import pandas as pd
import numpy as np
from ..misc import sort_data

# -----------------------------------------------------------------------------
#
# Functions and classes         test implemented?
#
# get_turning_points            yes
# filter_turning_points         yes
# count_turning_points          yes
# response_time                 yes
#
# -----------------------------------------------------------------------------


@sort_data
def get_turning_points(df):
    """
    Accepts:

        a Pandas dataframe of shape:

                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.

    Searches the pandas dataframe to isolate the obmt times of the
    turning points and create a dataframe of these points and their
    characteristics.

    Returns:

        a reduced Pandas dataframe of shape:

                obmt    rate    w1_rate turning
            1.  float   float   float   True

        or equivalent.
    """
    working_df = df.copy()

    # At a turning point, the differences either side differ in sign.
    # Therefore, create two arrays and test where the sign differences
    # are. Return the indices where this is the case.

    differences_lower = np.sign([*np.diff(working_df['rate'] -
                                 working_df['w1_rate']), 0])

    differences_upper = np.sign([1, *np.diff(working_df['rate'] -
                                 working_df['w1_rate'])])

    # If the first value is 0 then it is a turning point.
    # Else check for inverse.
    turning_points = [d1 == -d2 if d1 != 0 else True for d1, d2 in
                      zip(differences_lower, differences_upper)]

    # Setting the last value of differences_lower==0 ensures it will
    # always return a false positive. Rectify that here. In reality you cannot
    # meaningfully ever consider the last datapoint to be a turning point so
    # this is also okay from that perspective.
    turning_points[-1] = False

    working_df['turning'] = turning_points
    return working_df[working_df['turning']]


def filter_turning_points(df, threshold=1):
    """
    Accepts:

        a Pandas dataframe of shape:

                obmt    rate    w1_rate (turning)
            1.  float   float   float   (True   )

        or equivalent. If the turning column is not present, indicating
        the dataframe not having already had turning points found,
        get_turning_points() is run on it first.

    By observing the difference in (rate - w1_rate) between
    neighbouring turning points, determines which are significant.

    Kwargs:

        threshold (float, default=1):
            the threshold for the difference in (rate - w1_rate) for two
            turning points above which both are considered significant.

    Returns:

        a Pandas dataframe of shape:

                obmt    rate    w1_rate turning
            1.  float   float   float   True

        or equivalent, of the locations of the significant turning
        points.
    """

    # Check if the dataframe received has already had
    if 'turning' in df.columns:
        working_df = df.copy()
    else:
        working_df = get_turning_points(df)

    # Relevant turning points have differences in amplitude > threshold.
    # Use this to isolate relevant points from oscillatory noise.
    turning_points = [True if abs(diff) > threshold else False for diff in
                      [0, *np.diff(working_df['rate'] -
                       working_df['w1_rate'])]]

    working_df['turning'] = turning_points

    return working_df[working_df['turning']]


def count_turning_points(df, threshold=1):
    """
    Accepts:

        a Pandas dataframe of shape:

                obmt    rate    w1_rate
            1.  float   float   float

        or equivalent.

    This should be a hit detected using
    hits.response.anomaly.isolate_anomaly().

    Counts the number of turning points with magnitude greater than
    threshold.

    Kwargs:

        threshold (float, default=1):
            the threshold for the difference in (rate - w1_rate) for two
            turning points above which both are considered significant.

    Returns:

        a tuple of:

            the magnitude of the hit, the number of turning points in
            the response to the hit.
    """
    turning_points = filter_turning_points(get_turning_points(df),
                                           threshold=threshold)

    peak = max(abs(df['rate'] - df['w1_rate']))

    return (peak, len(turning_points))


@sort_data
def response_time(df, t=10, window_size=25):
    """
    Accepts:

        a Pandas dataframe of shape:

                obmt    rate    w1_rate hits
            1.  float   float   float   bool
        or equivalent.

    This should be a hit detected using
    hits.response.anomaly.isolate_anomaly().

    By comparing the baseline level before the hit with the baseline
    level after the hit, is able to calculate the time taken for the
    satellite to fully recover from the hit's effects.

    Kwargs:

        t (float, default=10):
            the amount of samples before the peak of the hit before
            which the rate can be considered to be the baseline.

        window_size (float, default=25):
            samples to be considered in the rolling window.

    Returns:

        a Pandas dataframe of shape:

            start_obmt  end_obmt
        1.  float       float
    """
    # First time when the hit is marked as such.
    hit_start = min(df[df['hits']]['obmt'])

    peak = max(abs(df['rate'] - df['w1_rate']))  # Height of the hit.

    working_df = df.copy()

    # This calculates the base level of noise. It takes the maximum
    # displacement of the rate from 0 from the start of the event to t
    # samples before hit_start.
    base_level = max([abs(x) for x in (working_df['rate'] -
                      working_df['w1_rate'])[working_df['obmt'] <
                      hit_start].tolist()[:-t]])

    # Create a window of window_size samples to move along the axis,
    # checking if the maximum value within the window is less than the
    # baseline maximum. Once it is less, the hit is over.

    window = (working_df['rate'] -
              working_df['w1_rate'])[working_df['obmt'] >=
                                     hit_start].tolist()[:window_size]
    i = 1

    while max([abs(x) for x in window]) > base_level:
        window = (working_df['rate'] -
                  working_df['w1_rate'])[working_df['obmt'] >=
                                         hit_start][i:].tolist()[:window_size]
        i += 1

    # Return the time from the start of the window within which the hit
    # ends - hit_start
    return (peak, working_df['obmt'][working_df['obmt'] >=
                                     hit_start].tolist()[i] - hit_start)
