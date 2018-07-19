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
        self._cutoff = 3e-8
        self._alpha = 1 - np.exp(-1 * self._dt * self._cutoff)
#-----------------------------------------------------------------------------

#public methods --------------------------------------------------------------
    def tweak_cutoff(self, cutoff):
        """Change the cutoff frequency for the lowpass filter."""
        self._cutoff = cutoff
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
                x = alpha * (data_array[i] - x)
            except(IndexError):
                break
            yield x
            i += 1

    _filter = _low_pass
#-----------------------------------------------------------------------------
