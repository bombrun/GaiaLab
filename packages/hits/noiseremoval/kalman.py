"""
 Kalman filter implementation in Python

 The Kalman filter is an algorithm for the prediction of the true state of a noisy system.
 It works through prediction and measurement cycles wherein the previous state of the 
 system is considered.

 The key equations for the Kalman filter are:
        
    -----SCALAR VERSION-----------
    
    time update---------------------------------------------------------->
    x = A * x + B * u

    P_pri = A * P_pos * A + Q
    --------------------------------------------------------------------->
    
    measurement update--------------------------------------------------->

    K =   P_pri
        (P_pri + R) 

    P_pos = (I - K) * P_pri

    x_pos = x_pri + K * (z - H * x_pri)
    --------------------------------------------------------------------->


    -----MATRIX VERSION-----------
    where @ is the matrix multiplication operator:

    time update---------------------------------------------------------->
    x = A @ x + B @ u

    P_pri = A @ P_pos @ A + Q
    --------------------------------------------------------------------->
    
    measurement update--------------------------------------------------->
    e_pri = x - x_pri

    e_pos = x - x_pos

    K is equal to P_pri @ H_T @ (H @ P_pri @ H_T + R) ^ (-1) 
    inverse matrices are often numerically unstable therefore
    formulate as Alpha K = Beta and solve algorithmically
    for K.
    
    Alpha = H @ P_pri @ H_T + R

    Beta = P_pri @ H_T

    P_pos = (I - K @ H) @ P_pri

    x_pos = x_pri + K @ (z - H @ x_pri)
    --------------------------------------------------------------------->

 It is implemented here as a class to accept measured data and apply the Kalman filter.
 This returns a generator allowing large datasets to be used without being committed to memory twice.
 
 Current implementation supports 1-dimensional, constant data. 

 Toby James 2018
"""
import numpy as np
import pandas as pd
from array import array

#TODO implement extended Kalman and use to identify decay pattern for hits (assume exponential?)
#TODO can you transform detected (and Kalman-cleaned?) hits into some sort of linear space to get actual decaying exponential coefficents using basic Kalman??
#                                       answer: maybe??
#TODO decide on appropriate starting values for q and r.
#TODO implement multiple dimensional Kalman.

class KalmanData():
    """
    Data with Kalman filter applied.
    """
    def __init__(self, data_array, save=False):
        """
        Accepts:
            
            An array.

        Can be initialised with any iterable array of real numerical values.
        
        Kwargs:
            
            save(bool, default=False):
                If True, saves the Kalman-filtered data to self.kalman_data on initialisation.
                If False, doesn't. This saves memory.
        """
        if  hasattr(data_array, "__iter__"):

            if isinstance(data_array, pd.DataFrame):
                self._from_pandas(data_array)

            elif all(isinstance(data_point, (int, float)) for data_point in data_array):
                self._data = array('d', data_array)

            else:
                for data_point in data_array:
                    if not isinstance(data_point, (int, float)):
                        bad_type = type(data_point)
                        break
                raise(TypeError("Data type %r is not supported. Please supply an iterable of numerical values." % bad_type))

        else:
            raise(TypeError("Data type %r is not supported. Please supply an iterable of numerical values." % type(data_array)))
        
        self._kalman_data = KalmanData._kalman(self._data)
        
        if save:
            self.kalman_data = list(KalmanData._kalman(self._data))

    def __len__(self):
        """Delegate to __len__ of array.array."""
        return len(self._data)

    def __repr__(self):
        """No benefit in showing all values for large objects. Especially since the most useful data in this class is generated on the fly."""
        return "Kalman data object, length: %r\n[%r ... %r]" % (len(self), self._data[0], self.data[-1])
    

    def __getitem__(self, index):
        """Apply Kalman filter to selected regions."""
        self._kalman_data = KalmanData._kalman(self.data[index])
        return self._data[index]
    
    # Comparison operators:
    #-------------------------------------
    def checkdatatype(func): # decorator to automatically return false for non-matching datatypes
        """Wrap verify as a decorator."""
        def verify(self, other):
            """Automatically returns False for comparison of two non-matching datatypes."""
            if isinstance(other,KalmanData):
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
    @checkdatatype
    def __ne__(self, other):
        return self._data != other._data
    #--------------------------------------

    def __add__(self, other):
        """Allows addition of scalars and (elementwise) arrays of equal length to self."""
        if isinstance(other, KalmanData):
            if len(self) == len(other):
                return KalmanData([a + b for a, b in zip(self._data, other._data)])
            else:
                raise(ValueError("Unable to broadcast together operands of shape %r and %r." % (len(self),len(other))))
        elif hasattr(other, "__iter__") and not isinstance(other, str):
            if len(self) == len(other):
                return KalmanData([a + b for a, b in zip(self._data, other)])
            else:
                raise(ValueError("Unable to broadcast together operands of shape %r and %r." % (len(self),len(other))))
        elif isinstance(other, (float, int)):
            return KalmanData([a + other for a in self._data])
        else:
            raise(TypeError("Unable to broadcast together operands of type %r and %r." % (type(self), type(other))))
    
    def __mul__(self, other):
        """Allows multiplication by scalars and elementwise by arrays of equal length to self."""
        if isinstance(other, (int, float)):
            return KalmanData([a * other for a in self._data])
        elif hasattr(other, "__iter__") and len(other) == len(self) and not isinstance(other, str):
            return KalmanData([a * b for a,b in zip(self, other)])
        else:
            raise(TypeError("Unable to multiply types %r and %r." % (type(self), type(other))))

    def __sub__(self, other):
        """Allows subtraction of scalars and (elementwise) arrays of equal length to self."""
        if hasattr(other, "__iter__") and not isinstance(other, str):
            return self + [(-1) * b for b in other]
        elif isinstance(other, (float, int, KalmanData)):
            return self + (-1) * other # addition and multiplication are defined so define subtraction like this.
        else:
            raise(TypeError("Unable to broadcast together operands of type %r and %r." % (type(self), type(other))))
    
    def __truediv__(self, other):
        """Allows division by scalars and elementwise by arrays of equal length to self.""" 
        if isinstance(other, (float, int)):
            return KalmanData([a/other for a in self._data])
        elif hasattr(other, "__iter__") and not isinstance(other, str) and len(self) == len(other):
            return KalmanData([a / b for a,b in zip(self._data, other)])
        elif isinstance(other, KalmanData) and len(self) == len(other):
            return KalmanData([a / b for a,b in zip(self._data, other._data)])
        else:
            raise(TypeError("Unable to divide type %r by type %r." % (type(self), type(other))))

    def __call__(self): # allow easy iteration over generated data.
        """Returns a generator for the Kalman filtered data."""
        return self._kalman_data

    def _from_pandas(self, df): # I wanted this to be a separate function called by __init__. Not sure why, but I think it is sufficiently unique to justify existing in its own right.
        """Create KalmanData object from pandas dataframe."""
        if 'rate' and 'w1_rate' in df.columns:
            data_array = df['rate'] - df['w1_rate']
            self._data = data_array
        elif 'rate' in df.columns:
            data_array = df['rate']
            self._data = data_array
        else:
            raise(ValueError("The dataframe used does not have the correct columns to use this method. Please manually prepare the data and initialise the class the normal way."))
    
    def to_pandas(self, obmt=None, columns=None, w1_rate=None):
        """
        Accepts:
            
            Nothing.

        Returns a pandas dataframe of the Kalman filtered data.

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
        if obmt is None:
            obmt = range(len(self))
        if w1_rate is not None:
            rate = self.kalman_data - w1_rate
            df = pd.DataFrame(data=dict(obmt = obmt,
                                          rate = rate,
                                          w1_rate = w1_rate))
        else:
            rate = self.kalman_data
            df = pd.DataFrame(data=dict(obmt = obmt,
                                          rate = rate))
            df['w1_rate'] = df['rate'].copy().rolling(window=3600, min_periods=0).mean()
        return df
        

    q = None
    r = None
    
    def save(self):
        """Saves the Kalman filtered data to a variable - self.kalman_data."""
        self.kalman_data = list(self._kalman_data)

    def reset(self):
        """Resets the generator."""
        self._kalman_data = KalmanData._kalman(self._data)
    
    def tweak_q(self, q):
        """Change the variance of the noise on the measurable value."""
        self.q = q
        self.reset()

    def tweak_r(self, r):
        """Change the variance of the noise on the measured value."""
        self.r = r
        self.reset()
    
    @staticmethod
    def _var(data_array, samples):
        """Variance of the data."""
        return np.var(data_array[0:samples])

    @staticmethod
    def _kalman(data_array, r=r, q=q, samples=50):
        """
        Accepts:
            
            An array.

        Accepts any array of numerical, real values.

        Performs the Kalman filter algorithm on the calculated data.

        Is a generator rather than a function. self._kalman_data is the generator applied to self._data.

        Kwargs:
            
            r (float, default=1):
                The variance of the noise on the measured data. Changeable through self.tweak_r().
            
            q (float, default is calculated as the variance of the first samples=samples.):
                The variance of the noise on the measurable data. Changeable through self.tweak_q().

            samples: (float, default=50):
                The number of samples to be used to calculated the variance on the data.
        
        Yields:
            
            The next value as predicted by the Kalman filter algorithm.
        """
        x = data_array[0]
        p = x**2
        if q is None:
            q = KalmanData._var(data_array, samples)
        i = 0
        while True:
            p = p + q
            try:
                z = data_array[i]
            except(IndexError):
                break
            if r is None:
                r = 1
            K = p/(p+r)
            x = x + K * (z - x)
            p = (1-K)*p
            yield x
            i += 1
