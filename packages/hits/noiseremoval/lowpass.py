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
try:
    import filters
except(ImportError):
    import hits.noiseremoval.filters as filters

class LowPassData(filters.FilterData):
    """
    Low pass filter implementation.
    """

#special methods--------------------------------------------------------------
    def __init__(self, *args):
        filters.FilterData.__init__(self, *args)

        #set up private variables:
        if self._obmt is not None:
            self._dt = (self._obmt[1] - self._obmt[0]) * 21600 #convert to seconds
        else:
            self._dt = 1
        self._cutoff = 0.05 #this is a reasonable cutoff frequency to default to
        self._alpha = (2 * np.pi * self._dt * self._cutoff)/(2 * np.pi * self._dt * self._cutoff + 1)
#-----------------------------------------------------------------------------

#public methods --------------------------------------------------------------
    def tweak_cutoff(self, cutoff):
        """Change the cutoff frequency for the lowpass filter."""
        self._cutoff = cutoff
        self._alpha = 1 - np.exp(-1 * self._dt * self._cutoff)# (2 * np.pi * self._dt * self._cutoff)/(2 * np.pi * self._dt * self._cutoff + 1)
        self.reset()
#-----------------------------------------------------------------------------

#private methods--------------------------------------------------------------    
    def _low_pass(self, data_array, alpha=None):
        if alpha == None:
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

    _filter = _low_pass
#-----------------------------------------------------------------------------
