"""
Low pass filter implementation in python.

The low pass filter is defined by the recurrence relation:

    y_(n+1) = y_n + alpha (x_n - y_n)

where x is the measured data and y is the filtered data. Alpha is a constant
dependent on the cutoff frequency, f, and is defined as:

    alpha = 2 pi dt f
            2 pi dt f + 1

where dt is the time step used.
"""

import numpy as np
import scipy
import pandas as pd
from . import filter_base
from ..misc import s2o, o2s


class LowPassData(filter_base.FilterData):
    """
    Low pass filter implementation.
    """
    name = "LowPassData"

# Special methods--------------------------------------------------------------
    def __init__(self, *args, cutoff=None):
        filter_base.FilterData.__init__(self, *args)

        # Set up private variables.
        if self._obmt is not None:
            self._dt = o2s(self._obmt[1] - self._obmt[0])
        else:
            self._dt = 1
        if isinstance(cutoff, (float, int)):
            self._cutoff = cutoff
        else:
            print(self._data)
            self._cutoff = self._get_frequency_from_psd(self._data)

        self._alpha = (2 * np.pi * self._dt * self._cutoff)/(2 * np.pi *
                                                             self._dt *
                                                             self._cutoff + 1)
# -----------------------------------------------------------------------------

# Public methods --------------------------------------------------------------
    def tweak_cutoff(self, cutoff):
        """Change the cutoff frequency for the lowpass filter."""
        self._cutoff = cutoff
        self._alpha = 1 - np.exp(-1 * self._dt * self._cutoff)
        self.reset()
# -----------------------------------------------------------------------------

# Private methods--------------------------------------------------------------
    def _low_pass(self, data_array, alpha=None):
        if alpha is None:
            alpha = self._alpha

        x = data_array[0]
        i = 0
        while True:
            try:
                x += alpha * (data_array[i] - x)
            except(IndexError):
                break
            yield x
            i += 1

    def _get_frequency_from_psd(data):
        # Caluculate the sampling rate of the data. Since there are
        # occasionally jumps, it is worth checking that two random intervals
        # are the same to avoid accidentally calculating an incorrect rate.

        f = self._dt ** (-1)

        d = pd.DataFrame()

        d['freqs'], d['psd'] = scipy.signal.welch(data, fs=f)

        return d[d['psd'] == max(d['psd'])]['freqs'].tolist()[0]

# Reassign _filter to _low_pass
    _filter = _low_pass
# -----------------------------------------------------------------------------
