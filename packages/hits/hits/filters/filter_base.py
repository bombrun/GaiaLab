import numpy as np
import pandas as pd
from array import array


class FilterData():
    """
    Data with filter applied.
    Subclass this and apply custom filter methods.
    """
    name = "FilterData"
    # ----special methods------------------------------------------------------

    def __init__(self, data_array, save=False):
        """
        Accepts:

            An array.

        Can be initialised with any iterable array of real numerical
        values.

        Kwargs:

            save(bool, default=False):
                If True, saves the filtered data to self.data on
                initialisation. If False, doesn't. This saves memory.
        """
        # Check if data was initialised from pandas dataframe.
        self._initialised_from_pandas = False
        # If data_array can be iterated over, iterate over it a
        # and add each element
        if hasattr(data_array, "__iter__"):

            if isinstance(data_array, pd.DataFrame):
                # This is delegated to self._from_pandas().
                self._from_pandas(data_array)
                self._initialised_from_pandas = True

            elif all(isinstance(data_point,
                                (int, float)) for data_point in data_array):
                # Check that each element is a real number.
                self._data = array('d', data_array)

            else:  # Isolate the issue with the data and raise TypeError.
                for data_point in data_array:
                    if not isinstance(data_point, (int, float)):
                        break
                raise(TypeError("Data type %r is not supported. Please supply"
                                " an iterable of numerical values." %
                                type(data_point)))

        else:
            raise(TypeError("Data type %r is not supported. Please supply an"
                            "iterable of numerical values." %
                            type(data_array)))

        # Set up filtered data generator.
        self._filter_data = self._filter(self._data)

        if save:
            # Saves data as a list rather than as a generator. More
            # resource heavy but can be useful nonetheless.
            self.data = array('d', self._filter(self._data))
        else:
            self.data = None

        # private variables----------------------------------------------------
        if not self._initialised_from_pandas:
            self._indices = None
            self._obmt = None
            self._w1_rate = None

        self._rev = False

    def __len__(self):  # Delegate to __len__ of array.array.
        """Length of the data."""
        return len(self._data)

    def __repr__(self):
        # No benefit in showing all values for large objects. Especially
        # since the most useful data in this class is generated on the
        # fly.
        """Representation of the data."""
        # self.name here represents the class variable name and not
        # an instance variable.
        return self.name + " object, length: %r\n[%r ... %r]" % (len(self),
                                                                 self._data[0],
                                                                 self._data[-1]
                                                                 )

    def __getitem__(self, index):
        """
        Apply filter to selected regions.
        Returns a list rather than a generator.
        """
        return array('d', self._filter(self._data[index]))

    # Comparison operators:
    # -------------------------------------
    def checkdatatype(func):
        # Decorator to automatically return False for non-matching
        # datatypes.
        """Wrap verify as a decorator."""
        def verify(self, other):
            """
            Automatically returns False for comparison of two
            non-matching datatypes. Compares matching datatypes as
            expected.
            """
            if isinstance(other, FilterData):
                return(func(self, other))
            else:
                return False
        return verify
    # --------------------------------------

    @checkdatatype
    def __eq__(self, other):
        return self._data == other._data

    @checkdatatype
    def __gt__(self, other):
        return self._data > other._data

    @checkdatatype
    def __ge__(self, other):
        return self._data >= other._data

    @checkdatatype
    def __lt__(self, other):
        return self._data < other._data

    @checkdatatype
    def __le__(self, other):
        return self._data <= other._data

    def __ne__(self, other):
        if isinstance(other, FilterData):
            return self._data != other._data
        else:
            return True
    # --------------------------------------

    def __add__(self, other):  # Addition is implemented elementwise.
        """
        Allows addition of scalars and (elementwise) arrays of equal
        length to self.
        """
        if isinstance(other, FilterData):
            if len(self) == len(other):
                return type(self)([a + b for a, b in zip(self._data,
                                                         other._data)])
            else:
                raise(ValueError("Unable to broadcast together operands of "
                                 "shape %r and %r." % (len(self), len(other))))

        elif hasattr(other, "__iter__") and not isinstance(other, str):
            if len(self) == len(other):
                return type(self)([a + b for a, b in zip(self._data, other)])
            else:
                raise(ValueError("Unable to broadcast together operands of "
                                 "shape %r and %r." % (len(self), len(other))))

        elif isinstance(other, (float, int)):
            return type(self)([a + other for a in self._data])

        else:
            raise(TypeError("Unable to broadcast together operands of type "
                            "%r and %r." % (type(self), type(other))))

    def __mul__(self, other):
        """
        Allows multiplication by scalars and elementwise by arrays of
        equal length to self.
        """
        if isinstance(other, (int, float)):
            return type(self)([a * other for a in self._data])

        elif isinstance(other, FilterData):
            return type(self)([a * b for a, b in zip(self._data, other._data)])

        elif hasattr(other, "__iter__") and len(other) == len(self) and not \
                isinstance(other, str):
            # All of these are necessary for elementwise multiplication.
            return type(self)([a * b for a, b in zip(self._data, other)])

        else:
            raise(TypeError("Unable to multiply types %r and %r."
                            % (type(self), type(other))))

    def __sub__(self, other):
        """
        Allows subtraction of scalars and (elementwise) arrays of equal length
        to self.
        """
        if hasattr(other, "__iter__") and not isinstance(other, str):
            return self + [(-1) * b for b in other]
        elif isinstance(other, (float, int, FilterData)):
            return self + (-1) * other
        else:
            raise(TypeError("Unable to broadcast together operands of type "
                            "%r and %r." % (type(self), type(other))))

    def __truediv__(self, other):  # Analogous to multiplication.
        """
        Allows division by scalars and elementwise by arrays of equal
        length to self.
        """
        if isinstance(other, (float, int)):
            return type(self)([a/other for a in self._data])

        elif hasattr(other, "__iter__") and not isinstance(other, str) and \
                len(self) == len(other):
            return type(self)([a / b for a, b in zip(self._data, other)])

        elif isinstance(other, FilterData) and len(self) == len(other):
            return type(self)([a / b for a, b in zip(self._data, other._data)])

        else:
            raise(TypeError("Unable to divide type %r by type %r."
                            % (type(self), type(other))))

    def __rmul__(self, other):
        # Commutativity is not an issue so simply swap the order of the
        # variables and multiply as defined under __mul__.
        return self * other

    def __call__(self):  # Allow easy iteration over generated data.
        """Returns a generator for the filtered data."""
        return next(self._filter_data)
    # -------------------------------------------------------------------------

    # Private methods and variables (so far as python allows)------------------

    def _from_pandas(self, df):
        """
        Create FilterData object from pandas dataframe.

        Since the obmt column of the pandas dataframe is not necessarily
        sorted, this is harder than creating from raw arrays as
        information on the order of the obmt and the indices needs to be
        stored for the algorithm to work and return a dataframe of the
        same shape as expected.
        """
        # obmt is not always increasing therefore if it is given, sort
        # the data according to obmt.
        if 'obmt' in df.columns:  # Saves obmt and indices to lists.
            sorted_df = df.sort_values('obmt')
            self._indices = array('d', sorted_df.index.values.tolist())
            self._obmt = array('d', sorted_df['obmt'].tolist())
        else:
            sorted_df = df

        # Save rate and w1_rate if both are given.
        if 'rate' and 'w1_rate' in sorted_df.columns:
            self._w1_rate = array('d', sorted_df['w1_rate'].tolist())
            data_array = (sorted_df['rate'] - sorted_df['w1_rate']).tolist()
            self._data = array('d', data_array)

        # If just rate is given, save rate.
        elif 'rate' in sorted_df.columns:
            data_array = sorted_df['rate'].tolist()
            self._data = array('d', data_array)
        # If rate is not given, guess at the correct columns to use.
        else:
            # Assumes the zeroth column is the time data and sets the
            # first column's data to be filtered.
            data_array = sorted_df[sorted_df.columns.values[1]]
            self._data = array('d', data_array)
            print("Dataframe not recognised. Indexing by column " +
                  str(sorted_df.columns.values[1]))

    @staticmethod
    def _var(data_array, samples):
        # Really trivial function, gets an estimate for the noise on the
        # data by observing the first samples.
        """Variance of the data."""
        return np.var(data_array[0:samples])

    @staticmethod
    def _filter(data_array):
        """
        Basic filter. Not a filter. Just a generator for the given data.
        """
        i = 0
        while True:
            try:
                yield data_array[i]
            except(IndexError):
                break
            i += 1

    # Public methods-----------------------------------------------------------
    def reset(self):
        """Resets the generator."""
        self._filter_data = self._filter(self._data)

    def reverse(self):
        """Reverse the order of the data to run the filter backwards."""
        self._data = array('d', reversed(self._data))
        self._rev = not self._rev

    def save(self):
        """Saves the filtered data to a variable - self.data."""
        # Reset first to ensure the generator starts from the first
        # datapoint.
        self.reset()
        self.data = array('d', self._filter_data)

    def to_pandas(self, obmt=None, columns=None, w1_rate=None):
        """
        Accepts:

            Nothing.

        Returns a pandas dataframe of the filtered data.

        Kwargs:

            obmt(array-like, default=None):
                Time series to match the data to. If none is given,
                defaults to an array of 1 second intervals.

            w1_rate(array-like, default=None):
                Windowed data. If none is given, it is created from the
                calculated data.

        Returns:

            A pandas dataframe of shape:

                    obmt    rate    w1_rate
                1.  float   float   float

        """
        # To return a dataframe, the data needs to exist first.
        self.save()

        if self._rev:
            self.data = array('d', reversed(self.data))

        if self._obmt is not None:
            # If obmt is already saved, then use this as the obmt
            # column.
            obmt = self._obmt

        elif obmt is None and self._obmt is None:
            # If it isn't, and there is no obmt given, just use an
            # array.
            obmt = np.linspace(0, len(self._data)-1, len(self._data))

        if w1_rate is not None:
            # If a w1_rate is given, calculate the rate from that.
                rate = self.data + w1_rate
        else:
            if self._w1_rate is not None:
                w1_rate = self._w1_rate
                rate = [a + b for a, b in zip(self.data, w1_rate)]
            else:
                # If not, just use the filtered data as the rate data,
                # set w1_data as None.
                rate = self.data

        if self._indices is not None:
            # If indices are saved, use these to index.
            indices = self._indices
        else:
            # If not, use an array.
            indices = np.linspace(0, len(self._data)-1, len(self._data))

        # Generate dataframe.
        sorted_df = pd.DataFrame(index=indices, data=dict(obmt=obmt,
                                                          rate=rate,
                                                          w1_rate=w1_rate))
        # If the initial data was created from a pandas dataframe, the
        # indices may now be out of order due to sorting by obmt.
        # Sort the dataframe by index to allow the data to be more
        # comparable with the original dataframe.
        df = sorted_df.sort_index()

        # If w1_rate was set as None, create new windowed data.
        if df['w1_rate'].values[0] is None:
            df['w1_rate'] = df['rate'].copy().rolling(window=3600,
                                                      min_periods=0).mean()

        return df

    def copy(self):
        return type(self)(self._data)
