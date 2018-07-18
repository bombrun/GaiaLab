import numpy as np
from functools import wraps
"""
Miscellaneous functions.
"""

def sort_data(func):
    """
    Initial datasets are rarely sorted by obmt.
    This is trivial to fix but needs to be fixed often.
    As per DRY, it is packaged here for use as a decorator.
    """
    @wraps(func)
    def sort(df, *args, **kwargs):
        sorted_df = df.sort_values('obmt')
        return func(sorted_df, *args, **kwargs)
    return sort
