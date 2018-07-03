# #Funtions to analyze AOCS response to hits
#
# Able to generate splines for the hits and plot them.
#
# Also able to isolate turning points in the hit region
# and use this information to determine suitable degrees
# of polynomial fit.


# Standard imports. Requires identifyClanks() from hitdetector.py.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
try:
    from hits.hitdetector import identifyClanks
except(ImportError):
    from hitdetector import identifyClanks
from scipy.interpolate import UnivariateSpline, BSpline


def isolateAnomalies(df, time_res=0.01):
    """
    Accepts:
        
        a Pandas dataframe of shape:
    
                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.

    Passes this dataframe to hits.hitdetector.identifyClanks()
    to identify the hits.

    Isolates these data.

    Kwargs:
        
        time_res (float, default=0.01):
            half the width of the neighbourhoods to be generated.

    Returns:

        a tuple of reduced Pandas dataframes of shape:

                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.
    """

    working_df = identifyClanks(df)[0] #calls identifyClanks() from hitdetector to identify clanks and hits
    hit_df = working_df[working_df['hits']] #isolate the regions where hits are detected
    
    #Generate neighbourhoods around the hits with width 2*time_res
    hit_neighbourhoods = [working_df[abs(working_df['obmt'] - time) < time_res] for time in hit_df['obmt']]

    return tuple(hit_neighbourhoods)


def splineAnomalies(df, smooth=0.5, plot=False, B=False, turning=False, filtered=False):   #This is currently nothing but scipy's UnivariateSpline class.
                                                                           #A custom implementation may yield benefits, however scipy's splining
                                                                           #algorigthms are incredibly fast.
    """
    Accepts:
        
        a Pandas dataframe of shape:
    
                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.

    This dataframe should be a hit neighbourhood as generated by 
    isolateAnomalies(). The function will work on larger datasets
    but there is nothing to be gained by splining these - and the 
    time taken to do so would be significantly larger.

    Runs scipy's splining algorithms on the hit neighbourhoods
    to generate a spline to fit the data.

    Kwargs:

        smooth (float, default=0.5):
            smoothing factor for the generated splines.

        plot (bool, default=False): 
            if True, generates a plot of the data and the spline.
        
        B (bool, default=False): 
            if True, generates B splines.

        turning (bool, default=False):
            if True, plots all the turning points alongside a normal plot.

        filtered (bool, default=False):
            if True, plots the filterd turning points alongside a normal plot.

    Returns:
        
        tuple of the arrays of knots and coefficients (in that order)
        of the fitted spline.
    """

    sorted_hit = df.sort_values('obmt')            #independent data must be strictly increasing for scipy

    #create spline
    spl = UnivariateSpline(sorted_hit['obmt'], sorted_hit['rate'] - sorted_hit['w1_rate'])
    spl.set_smoothing_factor(smooth)
    knots, coeffs = (spl.get_knots(), spl.get_coeffs())

    xs = np.linspace(sorted_hit['obmt'].tolist()[0], sorted_hit['obmt'].tolist()[-1], 10000)

    if B:
        spl = BSpline(knots, coeffs, 3)


    #plot original data and spline
    if plot or turning or filtered:
        plt.scatter(df['obmt'], df['rate'] - df['w1_rate'])
        plt.plot(xs, spl(xs))
    
        if filtered or turning: #calls getTurningPoints() to isolate the turning points of the function.
        
            if turning:
                turning_points = getTurningPoints(df)
            
            elif filtered:      #calls filterTurningPoints() to only return interesting output
                turning_points = filterTurningPoints(getTurningPoints(df))
            
            plt.scatter(turning_points['obmt'], turning_points['rate'] - turning_points['w1_rate'], color='red')

        plt.show()

    return (knots, coeffs)

def getTurningPoints(df):
    """
    Accepts:
        
        a Pandas dataframe of shape:
    
                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.
    
    Searches the pandas dataframe to isolate the
    obmt times of the turning points and create
    a dataframe of these points and their characteristics

    Returns:
        
        a reduced Pandas dataframe of shape:
    
                obmt    rate    w1_rate turning
            1.  float   float   float   True

        or equivalent.
    """
    working_df = df.copy()
   
    sorted_df = working_df.sort_values('obmt')

    #At a turning point, the differences either side differ in sign.
    #Therefore, create two arrays and test where the sign differences are.
    #Return the indices where this is the case.

    differences_lower = np.sign([*np.diff(sorted_df['rate']-sorted_df['w1_rate']), 0]) #np.sign() returns 1 or -1
    differences_upper = np.sign([1, *np.diff(sorted_df['rate']-sorted_df['w1_rate'])]) #or 0 for 0

    turning_points = [d1 == -d2 if d1 != 0 else True for d1, d2 in zip(differences_lower, differences_upper)] #if the first value is 0 then it is a turning point. else check for inverse.

    turning_points[-1] = False #setting the last value of differences_lower==0 ensures it will always return a false positive. rectify that here.
                               #in reality cannot really ever consider the last datapoint to be a turning point so this is also okay from that
                               #perspective

    sorted_df['turning'] = turning_points
    return sorted_df[sorted_df['turning']]


def filterTurningPoints(df, threshold=1):
    """
    Accepts:
        
        a Pandas dataframe of shape:
    
                obmt    rate    w1_rate (turning)
            1.  float   float   float   (True   )

        or equivalent. If the turning column is not present,
        indicating the dataframe not having turning points
        found, getTurningPoints() is run on it first. 

        This should be a dataframe of turning point locations as 
        generated by getTurningPoints().

    By observing the difference in (rate - w1_rate)  between
    neighbouring turning points, determines which are significant.
    
    Kwargs:
        
        threshold (float, default=1):
            the threshold for the difference in (rate - w1_rate) for
            two turning points above which both are considered 
            significant.

    Returns:
        
        a Pandas dataframe of shape:
    
                obmt    rate    w1_rate turning
            1.  float   float   float   True
        
        or equivalent, of the locations of the significant turning 
        points.
    """

    if 'turning' in df.columns:#check if the dataframe received has already had
        working_df = df.copy() #turning points isolated or not
    else:
        working_df = getTurningPoints(df)
    
    #Relevant turning points have differences in amplitude > threshold.
    #Use this to isolate relevant points from oscillatory noise.
    turning_points = [True if abs(diff) > threshold else False for diff in [0,*np.diff(working_df['rate'] - working_df['w1_rate'])]]

    working_df['turning'] = turning_points

    return working_df[working_df['turning']]
