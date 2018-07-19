import numpy as np
import pandas as pd
from array import array


class FilterData():
    """
    Data with filter applied.
    Subclass this and apply custom filter methods.
    """
    #-----special methods-----------------------------------------------------
    def __init__(self, data_array, save=False):
        """
        Accepts:
            
            An array.

        Can be initialised with any iterable array of real numerical values.
        
        Kwargs:
            
            save(bool, default=False):
                If True, saves the filtered data to self.data on initialisation.
                If False, doesn't. This saves memory.
        """
        self._initialised_from_pandas = False #check if data was initialised from pandas dataframe
        if  hasattr(data_array, "__iter__"): # if data_array can be iterated over, iterate over it and add each element

            if isinstance(data_array, pd.DataFrame): # this is a more hefty task, delegated to self._from_pandas()
                self._from_pandas(data_array)
                self._initialised_from_pandas = True

            elif all(isinstance(data_point, (int, float)) for data_point in data_array): # check that each element is a real number
                self._data = array('d', data_array)

            else: # isolate the issue with the given array and raise a TypeError
                for data_point in data_array:
                    if not isinstance(data_point, (int, float)):
                        break
                raise(TypeError("Data type %r is not supported. Please supply an iterable of numerical values." % type(data_point)))

        else:
            raise(TypeError("Data type %r is not supported. Please supply an iterable of numerical values." % type(data_array)))
        
        self._filter_data = self._filter(self._data) # set up a generator within
        
        if save: # saves data as a list rather than as a generator. more resource heavy but useful nonetheless
            self.data = list(FilterData._filter(self._data))

        #private variables----------------------------------------------------
        if not self._initialised_from_pandas: #set equal to None if not initialised from pandas
            self._indices = None
            self._obmt = None
            self._w1_rate = None

        
    def __len__(self):#delegate to __len__ of array.array.
        """Length of the data."""
        return len(self._data)

    def __repr__(self): #No benefit in showing all values for large objects. Especially since the most useful data in this class is generated on the fly.
        """Representation of the data."""
        return "FilterData object, length: %r\n[%r ... %r]" % (len(self), self._data[0], self._data[-1])
    
    def __getitem__(self, index):
        """
        Apply filter to selected regions.
        Returns a list rather than a generator.
        """
        return list(self._filter(self._data[index]))
        
    # Comparison operators:
    #-------------------------------------
    def checkdatatype(func): # decorator to automatically return false for non-matching datatypes
        """Wrap verify as a decorator."""
        def verify(self, other):
            """Automatically returns False for comparison of two non-matching datatypes. Compares matching datatypes as expected."""
            if isinstance(other,FilterData):
                return(func(self, other))
            else:
                return False
        return verify
    #--------------------------------------
    @checkdatatype
    def __eq__(self, other):
        return self._data == other._data
    @checkdatatype
    def __gt__(self, other):
        return self._data >  other._data
    @checkdatatype
    def __ge__(self, other):
        return self._data >= other._data
    @checkdatatype
    def __lt__(self, other):
        return self._data <  other._data
    @checkdatatype
    def __le__(self, other):
        return self._data <= other._data

    def __ne__(self, other):
        if isinstance(other, FilterData):
            return self._data != other._data
        else:
            return True
    #--------------------------------------

    def __add__(self, other): #addition is implemented elementwise
        """Allows addition of scalars and (elementwise) arrays of equal length to self."""
        if isinstance(other, FilterData):#this is probably never going to be used but it would be bizarre for a class to refuse to combine with another instance of itself
            if len(self) == len(other):
                return FilterData([a + b for a, b in zip(self._data, other._data)])
            else:
                raise(ValueError("Unable to broadcast together operands of shape %r and %r." % (len(self),len(other))))

        elif hasattr(other, "__iter__") and not isinstance(other, str): #this is largely more likely: adding another array of the same length to the data
            if len(self) == len(other):
                return FilterData([a + b for a, b in zip(self._data, other)])
            else:
                raise(ValueError("Unable to broadcast together operands of shape %r and %r." % (len(self),len(other))))
        
        elif isinstance(other, (float, int)): # if adding a constant to the data, add it to every element
            return FilterData([a + other for a in self._data])
        
        else:
            raise(TypeError("Unable to broadcast together operands of type %r and %r." % (type(self), type(other))))
    
    def __mul__(self, other): # multiply by scalars and elementwise by arrays
        """Allows multiplication by scalars and elementwise by arrays of equal length to self."""
        if isinstance(other, (int, float)):
            return FilterData([a * other for a in self._data])
        elif isinstance(other, FilterData):
            return FilterData([a * b for a,b in zip(self._data, other._data)])
        elif hasattr(other, "__iter__") and len(other) == len(self) and not isinstance(other, str): # all of these are necessary for elementwise multiplication to make sense
            return FilterData([a * b for a,b in zip(self._data, other)])
        else:
            raise(TypeError("Unable to multiply types %r and %r." % (type(self), type(other))))

    def __sub__(self, other):
        """Allows subtraction of scalars and (elementwise) arrays of equal length to self."""
        if hasattr(other, "__iter__") and not isinstance(other, str):
            return self + [(-1) * b for b in other]
        elif isinstance(other, (float, int, FilterData)):
            return self + (-1) * other # addition and multiplication are defined so define subtraction like this.
        else:
            raise(TypeError("Unable to broadcast together operands of type %r and %r." % (type(self), type(other))))
    
    def __truediv__(self, other): # analogous to multiplication
        """Allows division by scalars and elementwise by arrays of equal length to self.""" 
        if isinstance(other, (float, int)):
            return FilterData([a/other for a in self._data])
        elif hasattr(other, "__iter__") and not isinstance(other, str) and len(self) == len(other):
            return FilterData([a / b for a,b in zip(self._data, other)])
        elif isinstance(other, FilterData) and len(self) == len(other):
            return FilterData([a / b for a,b in zip(self._data, other._data)])
        else:
            raise(TypeError("Unable to divide type %r by type %r." % (type(self), type(other))))

    def __rmul__(self,other): #commutativity is not an issue so simply swap the order of the variables and multiply as defined under __mul__
        return self*other

    def __call__(self): # allow easy iteration over generated data.
        """Returns a generator for the filtered data."""
        return next(self._filter_data)
    #-------------------------------------------------------------------------

    #private methods and variables (so far as python allows)------------------

    def _from_pandas(self, df): # I wanted this to be a separate function called by __init__. I think it is sufficiently unique to justify existing in its own right.
        """
        Create FilterData object from pandas dataframe.

        Since the obmt column of the pandas dataframe is not necessarily sorted, this is harder than creating from raw arrays as 
        information on the order of the obmt and the indices needs to be stored for the algorithm to work and return a dataframe 
        of the same shape as expected.
        """
        #obmt is not always increasing therefore if it is given, sort the data according to obmt.
        if 'obmt' in df.columns:# saves obmt and indices to lists. this obviously takes up potentially unnecessary memory
            sorted_df = df.sort_values('obmt')
            self._indices = array('d', sorted_df.index.values.tolist())
            self._obmt = array('d',sorted_df['obmt'].tolist())
        else:
            sorted_df = df

        #if rate and windowed rate are given, good - it will behave as expected
        if 'rate' and 'w1_rate' in sorted_df.columns:
            self._w1_rate = array('d', sorted_df['w1_rate'].tolist())
            data_array = (sorted_df['rate'] - sorted_df['w1_rate']).tolist()
            self._data = array('d', data_array)
        #if just the rate is given it will attempt to work with that but depending on the origin of the data this will not necessarily work as well
        elif 'rate' in sorted_df.columns:
            data_array = sorted_df['rate'].tolist()
            self._data = array('d', data_array)
        #if rate is not given the dataset is unusable
        else:
            raise(ValueError("The dataframe used does not have the correct columns to use this method. Please manually prepare the data and initialise the class the normal way."))

    @staticmethod
    def _var(data_array, samples): #really trivial function, gets an estimate for the noise by observing the first samples
        """Variance of the data."""
        return np.var(data_array[0:samples])
    
    @staticmethod
    def _filter(data_array):
        """Basic filter. Not a filter. Just a generator for the given data."""
        i = 0
        while True:
            try:
                yield data_array[i]
            except(IndexError):
                break
            i+=1

    #public methods-----------------------------------------------------------
    def reset(self):
        """Resets the generator."""
        self._filter_data = self._filter(self._data)
    
    def save(self):
        """Saves the filtered data to a variable - self.data."""
        self.reset() # reset first to ensure the generator starts from the first datapoint
        self.data = list(self._filter_data)
    
    def to_pandas(self, obmt=None, columns=None, w1_rate=None):
        """
        Accepts:
            
            Nothing.

        Returns a pandas dataframe of the filtered data.

        Kwargs:

            obmt(array-like, default=None):
                Time series to match the data to. If none is given, defaults to an array of 1 second intervals.

            w1_rate(array-like, default=None):
                Windowed data. If none is given, it is created from the calculated data.

        Returns:
            
            A pandas dataframe of shape:
                
                    obmt    rate    w1_rate
                1.  float   float   float

        """
        self.save() # to return a dataframe, the data needs to exist first 
        
        if self._obmt is not None:
        #if obmt is already saved, then use this as the obmt column
            obmt=self._obmt
        elif obmt is None and self._obmt is None:
        #if it isn't, and there is no obmt given, just use an array
            obmt = np.linspace(0,len(self._data)-1, len(self._data))
       
        if w1_rate is not None:
        #if a w1_rate is given, calculate the rate from that
                rate = self.data + w1_rate
        else:
            if self._w1_rate is not None:
                w1_rate = self._w1_rate
                rate = [a + b for a, b in zip(self.data, w1_rate)]
            else:
        #if not, just use the filtered data as the rate data, set w1_data as None
                rate = self.data

        if self._indices is not None:
        #if indices are saved, use these to index
            indices = self._indices
        else:
        #if not, use an array
            indices = np.linspace(0,len(self._data)-1, len(self._data))

        #generate dataframe
        sorted_df = pd.DataFrame(index=indices, data=dict(obmt = obmt,
                                                          rate = rate,
                                                          w1_rate = w1_rate))
        #if the initial data was created from a pandas dataframe, the indices will now be out of order.
        #sort the dataframe by index to allow the data to be more comparable with the original dataframe.
        df = sorted_df.sort_index()

        #if w1_rate was set as None, create new windowed data
        if df['w1_rate'].values[0] is None:
            df['w1_rate'] = df['rate'].copy().rolling(window=3600, min_periods=0).mean()

        return df

    def copy(self):
        return type(self)(self._data)
