"""
Hit detector functions for AGIS 2.2 alongscan datasets.

Contains functions for the identification of hits, the separation of 
hits from noise, and for plotting and highlighting both hits and noise.
"""

# Standard imports - sys to accept command line file arguments.
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import warnings
from hits.misc import sort_data
from numba import jit
from array import array

# Functions and classes         test implemented?
#
# identify_through_magnitude    yes
# identify_through_gradient     yes
# Abuelmaatti                   yes
# point_density                 yes
# filter_through_response       no
# identify_noise                no (fix required)
# anomaly_density               no
# plot_anomaly                  no (probably unnecessary)


@sort_data
@jit
def identify_through_magnitude(df, anomaly_threshold=2):
    """
    Accepts:
    
        a Pandas dataframe of shape:

                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.

    Anomalies are defined as locations where the instantaneous rate is 
    more than anomaly_threshold (default is 2 mas/s) more than the 
    windowed rate for that region.

    By inspection, this definition catches most hits in real data, for 
    suitable values of anomaly_threshold, but is also sensitive to 
    noise.
    
    Therefore, this function suffices for basic hit detection but 
    requires refining. Clank and noise identification is handled by 
    later functions.

    Kwargs:
        
        anomaly_threshold (float, default=2):
            difference between rate and w1_rate (in mas/s) above which a
            region is identified as anomalous.

    Returns:
       
        a tuple of:

            a Pandas dataframe of shape:

                    obmt    rate    w1_rate anomaly
                1.  float   float   float   bool

            or equivalent.
        
            and a dataframe of shape:

                    obmt
                1.  float

            containing the times of detected anomalies.
    """
    
    working_df = df.copy()  # Be careful with python mutables.

    # Add a column to the dataframe with truth values for anomalies.
    working_df['anomaly'] = (abs(working_df['rate']- working_df['w1_rate']) >=\
                                                             anomaly_threshold)

    # == True is not needed but makes clear the selection occuring here.
    times   = np.array(working_df['obmt'][working_df['anomaly'] == True])
    indices = np.array(working_df.index[working_df['anomaly'] == True])

    # Floor the times*10 and then divide by 10. then drop duplicates to 
    # isolate points to within 1/10 of a revolution, a reasonable 
    # accuracy for hit individuality.
    
    anomaly_df = pd.DataFrame(index=indices, \
                              data=dict(obmt = np.floor(times*10)/10))

    return (working_df,anomaly_df.drop_duplicates(subset='obmt'))

@sort_data
def identify_through_gradient(df, gradient_threshold=1):
    """
    Accepts:
        
        a Pandas dataframe of shape:

                obmt    rate    w1_rate
           1.  float   float   float

        or equivalent.

    Identifies anomalies in the data by identifying regions where the
    instantaneous change in hit rate is larger than gradient_threshold. 
    Sensitive to smaller amplitude hits than identify_through_magnitude, but 
    highly sensitive to noise for low values of gradient_threshold.

    Kwargs:
        
        gradient_threshold (float, default=0.3):
            the threshold for rate change above which a region is 
            identified as anomalous.

    Returns:
       
        a tuple of:

            a Pandas dataframe of shape:

                    obmt    rate    w1_rate anomaly
                1.  float   float   float   bool

            or equivalent.
        
            and a dataframe of shape:

                    obmt
                1.  float

            containing the times of detected anomalies.
    """
    working_df = df.copy()

    working_df = working_df.sort_values('obmt')

    working_df['grad'] = [0,*np.diff(working_df['rate']-working_df['w1_rate'])]
    
    working_df['anomaly'] = (abs(working_df['grad'] >= gradient_threshold))
    
    # == True is not needed but makes clear the selection occuring here.
    times   = np.array(working_df['obmt'][working_df['anomaly'] == True])
    indices = np.array(working_df.index[working_df['anomaly'] == True])

    # Floor the times*10 and then divide by 10. then drop duplicates to 
    # isolate points to within 1/10 of a revolution, a reasonable 
    # accuracy for hit individuality.
    anomaly_df = pd.DataFrame(index=indices, \
                              data=dict(obmt = np.floor(times*20)/20))

    return (working_df,anomaly_df.drop_duplicates(subset='obmt'))



class Abuelmaatti:
    """
    Fourier coefficient calculation algorithm proposed by Muhammad Tahir
    Abuelma'atti [1].

    A function, f, can be approximated through the summation of harmonic
    functions in calculable proportions. This is an algorithm for the 
    calculation of these amplitudes from samples of data. No 
    consideration for the effects of error on points is made.

    A function, offset from 0 by z, can be represented as:

                       M
    f(x) = delta(0) + sum (  delta(m) * cos(2 pi m x/B)
                      m=1   +gamma(m) * sin(2 pi m x/B)  ) + z.
    
    Abuelma'atti gives a numerical algorithm for the calculation of the
    coefficients delta and gamma from sampled - experimental - data
    without the need for integration. It is therefore very fast without 
    the need to sacrifice accuracy, although it does not take into
    account the errors on measured points, instead assuming them all to 
    be exact and representative of the true state of the system at that
    time.
    
    [1] Abuelma'atti MT. A Simple Algorithm for Computing the Fourier 
        Spectrum of Experimentally Obtained Signals. Applied Mathematics
        and Computation. 1999;98;pp229-239.
    """
    def __init__(self, x, y):
        """
        Accepts:
            array of times, array of measurements.
        """
        self.x = x
        self.y = y
        self.y0 = self.y[0]
        self.y = [z - self.y0 for z in y]
        self.B = x[-1] - x[0]
        self.alpha = np.diff(y)/np.diff(x)

        self.delta_0 = 1/self.B * (0.5*self.x[2]*y[2] + 0.5*(self.x[-1] \
                       - self.x[-2])*self.y[-2] + sum([(self.x[s+1]-self.x[s])\
                       *self.y[s+1] - 0.5 * (self.x[s+1]-self.x[s])\
                       *(self.y[s+1]-self.y[s])\
                       for s in range(2, len(self.x)-2)]))
    
    def delta(self, m):
        """
        The coefficient of the mth cosine harmonic in the Fourier
        expansion of a function.
        """
        return -self.B/(2 * (m*np.pi)**2) * (self.alpha[1] - self.alpha[-1] \
            + sum([(self.alpha[s+1] - self.alpha[s]) * \
                   np.cos(2*m*np.pi*self.x[s+1]/self.B)\
                   for s in range(2, len(self.x)-2)]))

    def gamma(self, m):
        """
        The coefficient of the mth sine harmonic in the Fourier
        expansion of a function.
        """
        return -self.B/(2 * (m*np.pi)**2)\
            * sum([(self.alpha[s+1] - self.alpha[s]) * \
                   np.sin(2*m*np.pi*self.x[s+1]/self.B)\
                   for s in range(2, len(self.x)-2)])
    def lam(self, m):
        """
        The amplitude of the mth harmonic in the Fourier expansion of a
        function.
        """
        return np.sqrt(self.delta(m) ** 2 + self.gamma(m) ** 2)

    def phi(self, m):
        """
        The phase angle of the mth harmonic in the Fourier expansion of 
        a function.
        """
        return np.atan(self.gamma(m)/self.delta(m))

    def f(self,x, harmonics):
        """ 
        The function fitted to the data, as produced by the algorithm.
        harmonics gives the number of harmonics to be calculated.
        """
        return self.delta_0 + sum(self.delta(m) * np.cos(2*np.pi*m*x/self.B) \
                                + self.gamma(m) * np.sin(2*np.pi*m*x/self.B) \
                                  for m in range(1,harmonics+1)) + self.y0

# TODO: implement identification algorithm through changes in clank
# periodicity due to hits.

#@sort_data
#def identify_through_abuelmaatti(df):

def point_density(df):
    """
    Accepts:
    
        a Pandas dataframe of shape:

                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.
    
    Calcualates the cumulative density of points as a decreasing series
    in amplitude.

    Returns:
        
        a tuple of:
            
            the amplitude array, the cumulative number of points greater
            than each amplitude.
    """
    rate = array('d', df['rate']-df['w1_rate'])
    max_height = max(rate)
    _rate = array('d',)
    _height = array('d',)
    while max_height > 0:
        _height.append(max_height)
        _rate.append(len([1 for x in rate if x > max_height]))
        max_height -= 0.1
    return (_height[::-1], _rate)

@sort_data
@jit
def filter_through_response(df, threshold=2):
    """
    Accepts:
    
        a Pandas dataframe of shape:

                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.
    
    By iterating backwards across the data, is able to trace the smooth
    responses to hits and identify noisey data - that is, sudden jumps
    in rate without a corresponding AOCS response. Removes these data.

    Kwargs:
         
         threshold (float, default=2):
            the threshold for a jump considered abnormally large.

    Returns:
       
        a Pandas dataframe of shape:

                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.
    """
    rate_array = array('d',)
    working_df = df.iloc[::-1]
    _rate = 0
    _rate_diff= 0
    hit_array = array('i',)
    for time, rate in zip(working_df['obmt'], 
                    working_df['rate']-working_df['w1_rate']):
        
        _rate_diff = rate-_rate

        if abs(_rate_diff) > threshold and abs(rate) > abs(_rate):
            rate = _rate

        rate_array.append(rate)
        _rate = rate

    df['rate'] = rate_array[::-1] + df['w1_rate']

    return df

def identify_noise(df): 
    """
    Accepts:
        
        a Pandas dataframe of shape:

                obmt    rate    w1_rate
           1.  float   float   float

        or equivalent.

    Calls identify_through_magnitude() on the dataframe to identify the 
    hits.

    Checks the periodicity of the hits to identify noise - any two hits 
    occuring with period constant to within 0.1 revolutions are assumed 
    to be noise - it can be demonstrated that the probability of two
    genuine hits occurring within this timescale is vanishingly 
    small.[1]

    This method does however reject hits that occur in close temporal 
    proximity to noise. This is a non negligible consideration.

    The chances of the periodic method incorrectly ruling out hits is
    therefore low but noise with longer period, or aperiodic noise is 
    not detected through this method.

    Returns:
    
        a tuple of a dataframe of shape:

                obmt    rate    w1_rate anomaly hits
            1.  float   float   float   bool    bool

        and the Pandas time dataframe returned by 
        identify_through_magnitude()[1]. See 
        help(identify_through_magnitude) for more information.


    [1] From Lennart Lindegren's SAG--LL-030 technical note 
        (http://www.astro.lu.se/~lennart/Astrometry/TN/Gaia-LL-031-
        20000713-Effects-of- micrometeoroids-on-GAIA-attitude.pdf), the 
        rate of micrometeoroid impacts of mass greater than 1e-13 can be
        shown not to exceed 0.01 per second. This is equivalent to 216
        per revolution. The rate of micrometeoroid impacts of mass large
        enough to cause a disturbance > 2mas/s can be shown to be ~6e-8 
        per second, ie ~1e-3 per revolution.

        The hits follow a poisson distribution with these rates as the 
        rate parameter. The difference between hits therefore follows an
        exponential distribution with the same rate parameter. 

        The difference between two differences between three datapoints 
        is considered - small differences indicates periodicity.  This 
        is given by the difference between two independent, 
        exponentially distributed variables with the same rate
        parameter, it can be shown that the probability of the
        difference between the difference between two genuine hits being
        less than 0.1 revolutions is around 1e-4. This metric is
        therefore accurate to around 0.01% accuracy.
    """

    data,t = identify_through_magnitude(df)

    # To detect periodic noise, the difference between hits is 
    # calculated. If the difference between neighbouring differences is
    # small (indicating periodicity), the anomalies are considered to be 
    # noise.

    if len(t['obmt']) < 3:                           
    # If there are fewer than 3 data points, the difference between the 
    # differences does not exist. Furthermore, it is unrealistic that 
    # any of these 3 are not genuine hits. The dataframe is simply
    # altered to the expected return shape and returned as is.
        working_df = data.copy()                            
        working_df['hits'] = working_df['anomaly'].copy()   
                                                            
        hit_df = working_df.loc[t.index]

        return (hit_df, t)

    else:
        # Generate differences and differences between them.
        differences = np.diff(t['obmt'])

        differences2 = np.diff(differences)
        # time_data dataframe is indexed as the time-sorted dataset, but
        # includes columns for the time differences.
        time_data = pd.DataFrame(index=t.index, data=dict(ombt = t['obmt'],
                                           diff = [1,*differences],
                                           diff_diff = [1,1, *differences2])) 

        # Mark anomalies as hits only if
        time_data['hits'] = [False if (diff < 0.1 and diff1 < 0.1) or \
        (diff1 < 0.1 and diff2 < \
        0.1) else True for diff, diff1, diff2 in zip(time_data['diff_diff'],
        [*time_data['diff_diff'].tolist()[1:], 1],
        [*time_data['diff_diff'].tolist()[2:], 1, 1])]

        working_df = data.copy()
        
        # Mark all entries in the hits column of the returned dataframe
        # as False unless they have a value in hit_data. In that case, 
        # use that value.
        working_df['hits'] = np.array([time_data.loc[index]['hits'] if index in\
        time_data.index else False for index in np.array(working_df.index)])
        
        return (working_df, t)
    
def anomaly_density(df, method='response', window_size=3600, **kwargs):
    """
    Accepts:
    
        a Pandas dataframe of shape:

                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.

    By calling identify_through_magnitude on the data, calculates the 
    anomaly density for each region (in unique hits per datapoint).

    Kwargs:
        
        window_size (float, default=3600):
            the size of the window to be used to calculate the rolling
            anomaly density.

    Returns:
        
        a Pandas dataframe of shape:

                obmt    rate    w1_rate anomaly density
            1.  float   float   float   bool    float

        or equivalent.
    """

    
    if 'anomaly' not in df.columns.values:
        identify = method_dict[method]
        anomaly_df = identify(df)[0]
    else:
        anomaly_df = df.copy()

    anomaly_df['density'] = anomaly_df['anomaly'].rolling(window=window_size,
                                                  min_periods=0, 
                                                  **kwargs).mean()
    return anomaly_df


def plot_anomaly(*dfs, method='magnitude', highlight=False, highlights=False, 
                 show=True, **kwargs):
    """
    Accepts:
        
        Pandas dataframes of shape:

                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.

    Calls identify_through_magnitude() or identify_noise() on each 
    dataframe as appropriate. 

    identify_through_magnitude() (noise=False, default) is much faster 
    due to jit compilation.
    
    Plots (rate - w1_rate) against obmt.

    Kwargs:

        method (string, default="magnitude"):
            the method to use for hit identification.
            Options are:
                abuelmaatti
                magnitude
                gradient
                response

        highlight (bool, default=False);
        highlights (bool, default=False):
            if True,  plots windows of width 0.1 (= tolerance for hit
            quantisation) around hit locations.
            
            highlight and highlights are both acceptable parameters for 
            ease of use.

        show (bool, default=True):
            if True, shows the plot. If False, plt.show() needs to be 
            called after.

        **kwargs:
            passes these to plt.scatter().

    Returns:
        
        the Pandas dataframe of times generated by 
        identify_through_magnitude(). See 
        help(identify_through_magnitude) for more information.
    """
    try:
        identify = method_dict[method]
    except(KeyError):
        raise(KeyError("Unknown value given for kwarg 'method'."))
    
    for df in dfs:

        data,t = identify(df) 
        # Create dummy colour array where all are red.
        colors = pd.DataFrame(index=t.index.values, \
        data=dict(color = ['red' for time in t.index]))

        if highlight or highlights:
        # Get times of anomalies and corresponding colours.
            for index, row in t.iterrows():
                time = row['obmt']
                plt.axvspan(time, time+0.05, color=colors['color'][index], \
                            alpha=0.5) # Create coloured bars.
            plt.scatter(df.obmt, df.rate-df.w1_rate, s=1)
        else:
        # Basic plot.
            plt.scatter(df.obmt,df.rate-df.w1_rate,s=2)

        # Pretty plot up and show.
        plt.xlabel("obmt")
        plt.ylabel("rate - w1_rate")
   
    if show:
        plt.show()

# Dictionary to allow hit detection method selection.
method_dict = dict(magnitude = identify_through_magnitude,
                   gradient = identify_through_gradient)
#                   abuelmaatti = identify_through_abuelmaatti,

if __name__ == '__main__':

    """
    File can be run from the command line or imported.
    If run from the command line, and passed files as input arguments,
    runs plot_anomaly() on the files given.
    """

    for datafile in sys.argv[1:]:
        df = pd.read_csv(datafile)
        hit_locs, _ = identify_noise(df)
        print(len(hit_locs[hit_locs['hits']]))
