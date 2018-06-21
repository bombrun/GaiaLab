# # Hit detector functions for datasets.
#
# Functions to identify sudden changes in rate.
#
# Toby James - 2018
# Adapted from previous rate analysis code by Alex Bombrun


#Standard imports - sys to accept command line file arguments
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import warnings
from numba import jit

@jit()
def identifyAnomaly(df, times=False):
    """
    Accepts a Pandas dataframe of the following shape

        obmt    rate    w1_rate
    1.  float   float   float

    or equivalent.

    Hits are defined as locations where the instantaneous rate is more
    than double the windowed - average - rate for that region.

    This definition catches all hits but is also sensitive to clanks.
    
    Therefore, this function suffices for basic hit detection but requires
    refining. Clank identification is handled by later functions.

    Normally the function returns the initial frame it is passed,
    but with a new column 'anomaly', with boolean values indicative
    of whether the detected rate is likely a anomaly or not.

    Return dataframe:

        obmt    rate    w1_rate anomaly
    1.  float   float   float   bool

    or equivalent.
     
    If the kwarg times=True is given, the function returns a tuple
    of the standard return dataframe and a dataframe of the form

        obmt
    1.  float

    containing the times of detected anomalies.
    """
    
    working_df = df.copy()

    working_df['anomaly'] = (abs(working_df['rate']- working_df['w1_rate']) >= 2) #adds a column to the dataframe with truth values for anomalies.

    if times:
        #== True is not needed but makes clearer the selection occuring here
        times   = np.array(working_df['obmt'][working_df['anomaly'] == True]) #array of times of anomalies
        indices = np.array(working_df.index[working_df['anomaly'] == True]) #array of indices of anomalies 

        anomaly_df = pd.DataFrame(index=indices, data=dict(obmt = np.floor(times*10)/10))
        return (working_df,anomaly_df.drop_duplicates(subset='obmt'))

    else:
        return working_df

def plotAnomaly(df, highlight=False, clanks=False):
    """
    Accepts a Pandas dataframe of the following shape

        obmt    rate    w1_rate
    1.  float   float   float

    or equivalent.

    Calls identifyAnomaly() or identifyClanks() on the dataframe as
    appropriate. (determined by the clanks flag.)

    identifyAnomaly() (clanks=False, default) is much faster due to jit compilation.

    If highlight=True is set, plots a graph of hits with windows of width
    0.1 (= tolerance for hit quantisation) around hit locations.

    If clanks=True is set, highlights the clanks in red and the hits
    in green.

    If neither are set, simply plots (rate - w1_rate) against obmt.
    """
    #spelling of colour is standardised to color, even if colour is used in comments
    if clanks:
    #Call identifyClanks() to locate hits and clanks, and colour code appropriately
        data,t = identifyClanks(df)
        colors = pd.DataFrame(index=t.index.values, data=dict(color = [(lambda x: 'green' if x else 'red')(data['hits'][time]) for time in t.index]))
    else:
    #Call identifyAnomaly() to locate hits
        data,t = identifyAnomaly(df, times=True)
        colors = pd.DataFrame(index=t.index.values, data=dict(color = ['red' for time in t.index]))
    
    if highlight:
    #get times of anomalies and corresponding colours
        for index, row in t.iterrows():
            time = row['obmt']
            plt.axvspan(time, time+0.1, color=colors['color'][index], alpha=0.5)
        plt.scatter(df.obmt, df.rate-df.w1_rate, s=0.1)
    else:
    #basic plot
        plt.scatter(df.obmt,df.rate-df.w1_rate,s=1)

    plt.xlabel("obmt")
    plt.ylabel("rate - w1_rate")
    plt.show()
   
    return t 

def identifyClanks(df): #It was found that jit compilation offered negligible performance benefits for this function.
                        #The aesthetic benefits of pythonic unpacking and listcomps mean jit compilation is not used.
                        #Re-design of the function to a compilable function may be worth doing.
    """
    Accepts a Pandas dataframe of the following shape

        obmt    rate    w1_rate
    1.  float   float   float

    or equivalent.

    Calls identifyAnomaly() on the dataframe to identify the hits.

    Checks the periodicity of the hits to identify clanks - any two hits occuring within 0.1
    seconds of one another are assumed to be clanks - it can be demonstrated that the probability
    of two genuine hits occuring within this timescale is << 5e-7 [1].

    This method does however reject hits that occur in close temporal proximity to a clank - a non negligible consideration.

    The chances of the periodic method incorrectly ruling out hits is therefore low but
    clanking with longer period, or aperiodic clanking is not detected through this method,
    and clanking affects hit detection.

    Returns tuple of a dataframe of shape:

        obmt    rate    w1_rate anomaly hits
    1.  float   float   float   bool    bool

    and the time dataframe returned by identifyAnomaly(times=True).
    """

    data,t = identifyAnomaly(df, times=True)

    sorted_t = t.sort_values('obmt') #initial dataset is not necessarily indexed in order with obmt

    #to detect periodic clanking behaviour, the difference between hits is calculated.
    #If the difference between neighbouring differences is small (indicating periodicity)
    #the anomaly is considered a clank
    
    differences = np.diff(sorted_t['obmt'])

    differences2 = np.diff(differences)

    time_data = pd.DataFrame(index=sorted_t.index, data=dict(ombt = sorted_t['obmt'],
                                       diff = [1,*differences],
                                       diff_diff = [1,1, *differences2])) #arbitrarily large values for the first two numbers: 1 suffices
                                                                          #since clank behaviour has period << 1s
    hit_data = time_data.copy()
    hit_data['hits'] = [False if diff < 0.5 else True for diff in time_data['diff_diff']] #mark appropriate anomalies as hits

    working_df = data.copy()
    #mark all entries in the hits column of the returned dataframe as False unless they have a value in hit_data. In that case, use that value.
    working_df['hits'] = np.array([hit_data.loc[index]['hits'] if index in hit_data.index else False for index in np.array(working_df.index)])
    
    return (working_df, t)

if __name__ == '__main__':

    """
    File can be run from the command line or imported.
    If run from the command line, and passed files as input arguments,
    runs plotAnomaly() on the files given.
    """

    for datafile in sys.argv[1:]:
        df = pd.read_csv(datafile)
        times = plotAnomaly(df, highlight=True)
        print(str(len(times)) + " anomalies detected.")


#[1] From Lennart Lindegren's [SAG--LL-030 technical note](http://www.astro.lu.se/~lennart/Astrometry/TN/Gaia-LL-031-20000713-Effects-of-micrometeoroids-on-GAIA-attitude.pdf),
#    the rate of micrometeorite impacts of mass greater than 1e-13 can be shown not to exceed 0.01 per second.
#
#    Masses this small are in fact difficult to detect anyway, and the flux of micrometeorites drops sharply as mass increases.
#    Thus, modelling as a poisson distribution the impacts with a rate of 0.01 per second, the probability of 2 in 0.1s is 5e-7.
#
#    The true probability of two detectable hits occuring within this time period is in fact even smaller since this is calculated for masses greater than 1e-13,
#    the vast majority (by incidence) of which are undetectable anyway.
