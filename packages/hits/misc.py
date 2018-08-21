import numpy as np
from functools import wraps
import pandas as pd
from numba import jit
from array import array
"""
Miscellaneous functions.
"""


def sort_data(func):
    """
    Initial datasets are rarely sorted by obmt. This is trivial to fix
    but needs to be fixed often. As per DRY, it is packaged here for use
    as a decorator.
    """
    @wraps(func)
    def sort(df, *args, **kwargs):
        if isinstance(df, pd.DataFrame) and 'obmt' in df.columns.values:
            sorted_df = df.sort_values('obmt')
        else:
            sorted_df = df
        return func(sorted_df, *args, **kwargs)
    return sort


def s2o(secs):
    """Convert seconds to obmt."""
    return secs/21600


def o2s(obmt):
    """Convert obmt to seconds."""
    return obmt*21600


@jit(nopython=True)
def isolate_true(data):
    """Turn all Trues except the first into Falses in a run of Trues."""
    data_backwards = data[::-1]
    x = []
    for i in range(len(data) - 1):
        if data_backwards[i] and data_backwards[i+1]:
            x.append(0)
        else:
            x.append(data_backwards[i])
    x.append(data[0])
    return x[::-1]


def isolate_hit_df(df):
    """
    Isolate events by removing multiple True readings for the same hit.
    """
    df['hits'] = isolate_true(list(df['anomaly']))
    df['hits'] = df['hits'].astype('bool')
    return df


def hit_start_end_df(df):
    """Label hit start and end times."""
    df['start'] = isolate_true(list(df['anomaly']))
    df['end'] = isolate_true(list(df['anomaly'])[::-1])[::-1]
    df['start'] = df['start'].astype('bool')
    df['end'] = df['end'].astype('bool')
    return df
