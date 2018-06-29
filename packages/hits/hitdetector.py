# # Hit detector functions for datasets.
#
# Functions to identify sudden changes in rate.


#Standard imports - sys to accept command line file arguments
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import warnings
from numba import jit

@jit()
def identifyAnomaly(df, anomaly_threshold=2):
    """
    Accepts:
    
        a Pandas dataframe of shape:

                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.

    Anomalies are defined as locations where the instantaneous rate is more
    than anomaly_threshold (default is 2 mas/s) more than the windowed 
    rate for that region.

    By inspection, this definition catches most hits in real data but 
    is also sensitive to clanks.
    
    Therefore, this function suffices for basic hit detection but requires
    refining. Clank identification is handled by later functions.

    Normally the function returns the initial frame it is passed,
    but with a new column 'anomaly', with boolean values indicative
    of whether the detected rate is likely a anomaly or not.
    
    Kwargs:
        
        anomaly_threshold (float, default=2):
            difference between rate and w1_rate (in mas/s) above which

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
    
    Deprecation warning:
        times removed as kwarg and implemented as standard behaviour.
        No use case for identifyAnomaly(times=False).
    """
    
    working_df = df.copy()  #be careful with python mutables

    working_df['anomaly'] = (abs(working_df['rate']- working_df['w1_rate']) >= anomaly_threshold) #adds a column to the dataframe with truth values for anomalies.

    #== True is not needed but makes clearer the selection occuring here
    times   = np.array(working_df['obmt'][working_df['anomaly'] == True]) #array of times of anomalies
    indices = np.array(working_df.index[working_df['anomaly'] == True]) #array of indices of anomalies 

    #floor the times*10 and then divide by 10. then drop duplicates to isolate points to within 1/10 of a revolution, a reasonable accuracy
    #for hit individuality.
    anomaly_df = pd.DataFrame(index=indices, data=dict(obmt = np.floor(times*10)/10))

    return (working_df,anomaly_df.drop_duplicates(subset='obmt'))

def identifyClanks(df): #It was found that jit compilation offered negligible performance benefits for this function.
                        #The aesthetic benefits of pythonic unpacking and listcomps mean jit compilation is not used.
                        #Re-design of the function to a compilable function may be worth doing.
    """
    Accepts:
        
        a Pandas dataframe of shape:

                obmt    rate    w1_rate
           1.  float   float   float

        or equivalent.

    Calls identifyAnomaly() on the dataframe to identify the hits.

    Checks the periodicity of the hits to identify clanks - any two hits occuring with period
    constant to within 0.1 revolutions are assumed to be clanks - it can be demonstrated that 
    the probability of two genuine hits occuring within this timescale is vanishingly small.[1]

    This method does however reject hits that occur in close temporal proximity to a clank.
    This is a non negligible consideration.

    The chances of the periodic method incorrectly ruling out hits is therefore low but
    clanking with longer period, or aperiodic clanking is not detected through this method,
    and clanking affects hit detection.

    Returns:
    
        a tuple of a dataframe of shape:

                obmt    rate    w1_rate anomaly hits
            1.  float   float   float   bool    bool

        and the Pandas time dataframe returned by identifyAnomaly()[1].
        See help(identifyAnomaly) for more information.


    [1] From Lennart Lindegren's [SAG--LL-030 technical note](http://www.astro.lu.se/~lennart/Astrometry/TN/Gaia-LL-031-20000713-Effects-of-micrometeoroids-on-GAIA-attitude.pdf),
        the rate of micrometeorite impacts of mass greater than 1e-13 can be shown not to exceed 0.01 per second. This is equivalent to 216 per revolution.
        The rate of micrometeorite impacts of mass large enough to cause a disturbance > 2mas/s can be shown to be ~6e-8 per second, ie ~1e-3 per revolution.
        
        The hits follow a poisson distribution with these rates as the rate parameter. The difference between hits therefore follows an exponential distribution with
        the same rate parameter. 

        The difference between two differences between three datapoints is considered - small differences indicates periodicity.  This is given by thee difference between two independent,
        exponentially distributed variables with the same rate parameter, it can be shown that the probability of the difference between the difference between two genuine hits being
        less than 0.1 revolutions is around 1e-4. This metric is therefore accurate to around 0.01% accuracy.
    """

    data,t = identifyAnomaly(df)

    sorted_t = t.sort_values('obmt') #initial dataset is not necessarily indexed in order with obmt

    #to detect periodic clanking behaviour, the difference between hits is calculated.
    #If the difference between neighbouring differences is small (indicating periodicity)
    #the anomaly is considered a clank

    if len(sorted_t['obmt']) < 3:                           #if there are fewer than 3 data points, the difference between the differences
        working_df = data.copy()                            #does not exist. Furthermore, it is unrealistic that any of these 3 are not
        working_df['hits'] = working_df['anomaly'].copy()   #genuine hits. The dataframe is simply altered to the expected return shape and 
                                                            #returned as is.
        hit_df = working_df.loc[t.index]

        return (hit_df, t)

    else:

        differences = np.diff(sorted_t['obmt'])             #generate differences and differences of them

        differences2 = np.diff(differences)
        #time_data dataframe is indexed as the time-sorted dataset, but includes columns for the time differences.
        time_data = pd.DataFrame(index=sorted_t.index, data=dict(ombt = sorted_t['obmt'],
                                           diff = [1,*differences],
                                           diff_diff = [1,1, *differences2])) #arbitrarily large values for the first two numbers: 1 suffices
                                                                              #since clank behaviour has period << 1s
        hit_data = time_data.copy() #be careful with python mutables
        hit_data['hits'] = [False if diff < 0.5 else True for diff in time_data['diff_diff']] #mark appropriate anomalies as hits

        working_df = data.copy()    #see above^^
        #mark all entries in the hits column of the returned dataframe as False unless they have a value in hit_data. In that case, use that value.
        working_df['hits'] = np.array([hit_data.loc[index]['hits'] if index in hit_data.index else False for index in np.array(working_df.index)])
        
        return (working_df, t)


def plotAnomaly(df, highlight=False, clanks=False, **kwargs):
    """
    Accepts:
        
        a Pandas dataframe of shape:

                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.

    Calls identifyAnomaly() or identifyClanks() on the dataframe as
    appropriate. 

    identifyAnomaly() (clanks=False, default) is much faster due to jit compilation.
    
    Plots (rate - w1_rate) against obmt.

    Kwargs:

        highlight (bool, default=False):
            if True,  plots windows of width 0.1 (= tolerance for 
            hit quantisation) around hit locations.

        clanks (bool, default=False):
            if True, highlights the clanks in red and the hits
            in green.

        **kwargs:
            passes these to plt.scatter().

    Returns:
        
        the Pandas dataframe of times generated by identifyAnomaly().
        See help(identifyAnomaly) for more information.
    """
    #spelling of colour is standardised to color, even if colour is used in comments
    if clanks:
    #Call identifyClanks() to locate hits and clanks, and colour code appropriately
        data,t = identifyClanks(df)
        colors = pd.DataFrame(index=t.index.values, data=dict(color = [(lambda x: 'green' if x else 'red')(data['hits'][time]) for time in t.index]))
    else:
    #Call identifyAnomaly() to locate hits
        data,t = identifyAnomaly(df)
        colors = pd.DataFrame(index=t.index.values, data=dict(color = ['red' for time in t.index])) #create dummy colour array where all are red
    
    if highlight:
    #get times of anomalies and corresponding colours
        for index, row in t.iterrows():
            time = row['obmt']
            plt.axvspan(time, time+0.1, color=colors['color'][index], alpha=0.5) #create coloured bars
        plt.scatter(df.obmt, df.rate-df.w1_rate, s=0.1)
    else:
    #basic plot
        plt.scatter(df.obmt,df.rate-df.w1_rate,s=1)

    #pretty plot up and show
    plt.xlabel("obmt")
    plt.ylabel("rate - w1_rate")
    plt.show()
   
    return t #trivial return


if __name__ == '__main__':

    """
    File can be run from the command line or imported.
    If run from the command line, and passed files as input arguments,
    runs plotAnomaly() on the files given.
    """

    for datafile in sys.argv[1:]:
        df = pd.read_csv(datafile)
        hits, _ = identifyClanks(df)
        print(len(hits[hits['hits']]))
