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
import scipy.signal
from .misc import sort_data, isolate_hit_df, o2s, hit_start_end_df
from numba import jit
from array import array
from .filters import kalman as fk
from .filters import lowpass as fl

# -----------------------------------------------------------------------------
#
# Functions and classes         test implemented?
#
# identify_through_magnitude    yes
# identify_through_gradient     yes
# Abuelmaatti                   yes
# point_density                 yes
# filter_through_response       yes
# rms_diff                      yes
# stdev_diff                    yes
# rms                           yes
# stdev                         yes
# get_clank_frequency_from_psd  no
# anomaly_density               yes
# identify_hits                 no
# filter_and_identify           no
# null_identify                 no
# plot_anomaly                  no (probably unnecessary)
#
# -----------------------------------------------------------------------------


@sort_data
@jit
def identify_through_magnitude(df, threshold=2):
    """
    Accepts:

        a Pandas dataframe of shape:

                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.

    Anomalies are defined as locations where the instantaneous rate is
    more than threshold (default is 2 mas/s) more than the
    windowed rate for that region.

    By inspection, this definition catches most hits in real data, for
    suitable values of threshold, but is also sensitive to
    noise.

    Therefore, this function suffices for basic hit detection but
    requires refining. Clank and noise identification is handled by
    later functions.

    Kwargs:

        threshold (float, default=2):
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
    working_df['anomaly'] = (abs(working_df['rate'] -
                             working_df['w1_rate']) >=
                             threshold)

    times = np.array(working_df['obmt'][working_df['anomaly']])
    indices = np.array(working_df.index[working_df['anomaly']])

    # Floor the times*10 and then divide by 10. then drop duplicates to
    # isolate points to within 1/10 of a revolution, a reasonable
    # accuracy for hit individuality

    anomaly_df = pd.DataFrame(index=indices,
                              data=dict(obmt=np.floor(times*10)/10))

    return (working_df, anomaly_df.drop_duplicates(subset='obmt'))


@sort_data
def identify_through_gradient(df, threshold=0.3):
    """
    Accepts:

        a Pandas dataframe of shape:

                obmt    rate    w1_rate
           1.  float   float   float

        or equivalent.

    Identifies anomalies in the data by identifying regions where the
    instantaneous change in hit rate is larger than threshold.
    Sensitive to smaller amplitude hits than identify_through_magnitude,
    but highly sensitive to noise for low values of threshold.

    Kwargs:

        threshold (float, default=0.3):
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

    working_df['grad'] = [0, *np.diff(working_df['rate'] -
                          working_df['w1_rate'])]

    working_df['anomaly'] = (abs(working_df['grad'] >= threshold))

    times = np.array(working_df['obmt'][working_df['anomaly']])
    indices = np.array(working_df.index[working_df['anomaly']])

    # Floor the times*10 and then divide by 10. then drop duplicates to
    # isolate points to within 1/10 of a revolution, a reasonable
    # accuracy for hit individuality.
    anomaly_df = pd.DataFrame(index=indices,
                              data=dict(obmt=np.floor(times*20)/20))

    return (working_df, anomaly_df.drop_duplicates(subset='obmt'))


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

        self.delta_0 = 1/self.B * (0.5*self.x[2]*y[2] + 0.5*(self.x[-1] -
                                   self.x[-2])*self.y[-2] +
                                   sum([(self.x[s+1]-self.x[s]) *
                                        self.y[s+1] - 0.5 *
                                        (self.x[s+1]-self.x[s]) *
                                        (self.y[s+1]-self.y[s])
                                        for s in range(2, len(self.x)-2)]))

    def delta(self, m):
        """
        The coefficient of the mth cosine harmonic in the Fourier
        expansion of a function.
        """
        return -self.B/(2 * (m*np.pi)**2) * (self.alpha[1] - self.alpha[-1] +
                                             sum([(self.alpha[s+1] -
                                                 self.alpha[s]) *
                                                 np.cos(2 * m * np.pi *
                                                        self.x[s+1]/self.B)
                                                 for s in range(2,
                                                                len(self.x) -
                                                                2)]))

    def gamma(self, m):
        """
        The coefficient of the mth sine harmonic in the Fourier
        expansion of a function.
        """
        return -self.B/(2 * (m*np.pi)**2)\
            * sum([(self.alpha[s+1] - self.alpha[s]) *
                   np.sin(2*m*np.pi*self.x[s+1]/self.B)
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

    def f(self, x, harmonics):
        """
        The function fitted to the data, as produced by the algorithm.
        harmonics gives the number of harmonics to be calculated.
        """
        return self.delta_0 + sum(self.delta(m) * np.cos(2*np.pi*m*x/self.B) +
                                  self.gamma(m) * np.sin(2*np.pi*m*x/self.B)
                                  for m in range(1, harmonics+1)) + self.y0

# TODO: implement identification algorithm through changes in clank
# periodicity due to hits.


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
    _rate_diff = 0
    hit_array = array('i',)
    for time, rate in zip(working_df['obmt'],
                          working_df['rate'] - working_df['w1_rate']):

        _rate_diff = rate-_rate

        if abs(_rate_diff) > threshold and abs(rate) > abs(_rate):
            rate = _rate

        rate_array.append(rate)
        _rate = rate

    df['rate'] = rate_array[::-1] + df['w1_rate']

    return df


@sort_data
def rms_diff(df):  # Bizarrely, this is slower if @jit compiled.
    """Calculate the RMS value for the differences between points."""
    s = 0
    for diff in np.diff(df['rate'] - df['w1_rate']):
        s += diff**2

    return np.sqrt(s/(len(df['rate']) - 1))


@sort_data
def stdev_diff(df):
    """
    Calculate the standard deviation of the differences between
    points.
    """
    return np.std([abs(x)for x in np.diff(df['rate'] - df['w1_rate'])])


def rms(df):
    """Calculate RMS for a dataframe."""
    return np.sqrt(sum([x ** 2 for x in df['rate'] - df['w1_rate']]) /
                   len(df))


def stdev(df):
    """Calculate standard deviation of a dataframe."""
    return np.std(df['rate'] - df['w1_rate'])


def get_clank_frequency_from_psd(df):
    """
    Calculate clank period from power spectral density of a time-series.
    """
    # Caluculate the sampling rate of the data. Since there are
    # occasionally jumps, it is worth checking that two random intervals
    # are the same to avoid accidentally calculating an incorrect rate.

    f = o2s(np.mean(np.diff(df['obmt']))) ** (-1)

    d = pd.DataFrame()

    d['freqs'], d['psd'] = scipy.signal.welch(df['rate'] - df['w1_rate'], fs=f)

    return d[d['psd'] == max(d['psd'])]['freqs'].tolist()[0]/10


def anomaly_density(df, method='magnitude', window_size=3600, **kwargs):
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


def identify_hits(df):
    """
    Accepts:

        a Pandas dataframe of shape:

                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.

    Isolate hits identified by all detection methods.

    Returns:

        list of Truth values of hits corresponding to the given time
        series.
    """
    old_values = np.ones(len(df['obmt']))
    for key in method_dict.keys():
        working_df = method_dict[key](df)[0]

        hit_series = array('d', working_df['anomaly'])[::-1]
        new_series = array('d', [])

        for i in range(len(hit_series)):
            try:
                if hit_series[i+1]:
                    new_series.append(False)
                else:
                    new_series.append(hit_series[i])
            except(IndexError):
                new_series.append(False)
                break
        working_df['hits'] = new_series[::-1]
        new_values = [a and b for a, b in zip(old_values, new_series[::-1])]
        old_values = new_values

    return new_values


def filter_and_identify(df, method='magnitude', filter_threshold=None,
                        identify_threshold=None, kalman=True, lowpass=True):
    """
    Accepts:

        a Pandas dataframe of shape:

                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.

    Identifies hits in filtered data.

    Kwargs:

        method (string, default='magnitude'):
            Detection method to use for the anomaly identification.

        filter_threshold (float, default=None):
            Threshold for filtering. If None, threshold is calculated
            from the data.

        identify_threshold (float, default=None):
            Threshold for anomaly identification. If None, threshold is
            calculated from the data.

        kalman (bool, default=True):
            If True, applies a Kalman filter to the data before
            identifying anomalies.

        lowpass (bool, default=True):
            If True, applies a low-pass filter to the data before
            identifying anomalies.

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

    if filter_threshold is None:
        filter_threshold = 2

    working_df = filter_through_response(df, threshold=filter_threshold)

    if identify_threshold is None:
        identify_threshold = rms(working_df) + stdev(working_df)

    elif identify_threshold is None and method == "gradient":
        identify_threshold = rms_diff(working_df)

    if kalman:
        working_df = fk.KalmanData(working_df,
                                   q=stdev_diff(df) ** 2,
                                   r=stdev(df) ** 2).to_pandas()
    if lowpass:
        cutoff = get_clank_frequency_from_psd(df) / 10
        working_df = fl.LowPassData(working_df, cutoff=cutoff).to_pandas()

    identify = method_dict[method]

    return identify(working_df, threshold=identify_threshold)


def null_identify(df):
    """
    Accepts:

        a Pandas dataframe of shape:

                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.

    Does nothing to the data and returns it as given.

    Returns:

        the same dataframe, a single element list of [0].

    """
    return (df, pd.DataFrame(data=dict(obmt=[0, 0],
                                       rate=[0, 0],
                                       w1_rate=[0, 0])))


def plot_anomaly(*dfs, method=None, highlight=False, highlights=False,
                 show=True, line=False, **kwargs):
    """
    Accepts:

        Pandas dataframes of shape:

                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.

    Calls an identification fucntion on each dataframe as specified.

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
    if method is None:
        identify = null_identify
        kwarg_dict = dict()
    elif isinstance(method,
                    tuple) and isinstance(method[0],
                                          str) and isinstance(method[1],
                                                              dict):
        identify = method_dict[method[0]]
        kwarg_dict = method[1]
    else:
        try:
            identify = method_dict[method]
            kwarg_dict = dict()
        except(KeyError):
            raise(KeyError("Unknown value given for kwarg 'method'."))

    for df in dfs:

        data, t = identify(df, **kwarg_dict)
        # Create dummy colour array where all are red.
        colors = pd.DataFrame(index=t.index.values,
                              data=dict(color=['red' for time in t.index]))

        if highlight or highlights:
            # Get times of anomalies and corresponding colours.
            for index, row in t.iterrows():
                time = row['obmt']
                plt.axvspan(time, time+0.05, color=colors['color'][index],
                            alpha=0.5)  # Create coloured bars.
            plt.scatter(df.obmt, df.rate-df.w1_rate, s=1)
        elif line:
            i = 0
            isolated_data = isolate_hit_df(data)
            for time in isolated_data[isolated_data['hits']]['obmt']:
                if not i % 100:
                    print(i, "/", len(isolated_data[isolated_data['hits']]))
                plt.axvline(time, lw=1, color="orange")
                i += 1
            plt.scatter(data.obmt, data.rate-data.w1_rate, s=1)
        else:
            # Basic plot.
            plt.scatter(df.obmt, df.rate - df.w1_rate, s=2)

        # Pretty plot up and show.
        plt.xlabel("obmt")
        plt.ylabel("rate - w1_rate")

    if show:
        plt.show()


def log_start_and_end_times(*dfs, dest="hitranges.txt", method=None,
                            write_method='w+'):
    if method is None:
        identify = null_identify
        kwarg_dict = dict()
    elif isinstance(method,
                    tuple) and isinstance(method[0],
                                          str) and isinstance(method[1],
                                                              dict):
        identify = method_dict[method[0]]
        kwarg_dict = method[1]
    else:
        try:
            identify = method_dict[method]
            kwarg_dict = dict()
        except(KeyError):
            raise(KeyError("Unknown value given for kwarg 'method'."))

    for df in dfs:
        working_df = hit_start_end_df(identify(df, **kwarg_dict)[0])

        if dest is None:
            d = sys.stdout

        else:
            d = open(dest, write_method)

        for t0, t1 in zip(working_df[working_df['start']]['obmt'],
                          working_df[working_df['end']]['obmt']):

            d.write(str(t0) + "," + str(t1) + "\n")

        if d is not sys.stdout:
            d.close()


# Dictionary to allow hit detection method selection.
method_dict = dict(magnitude=identify_through_magnitude,
                   gradient=identify_through_gradient,
                   filtered=filter_and_identify)
#                   abuelmaatti = identify_through_abuelmaatti,
