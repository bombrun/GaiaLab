class Quaternion():
    """
    Quaternion class implemented to allow manipulation of quaternions with
    each other, matrices (including, where appropriate, vectors) and scalars.

    Declare a quaternion of the form q = w + xi + yj +zk by 

    >> q = Quaternion(w,x,y,z)

    Addition, subtraction, multiplication and division are all supported natively
    between quaternions and where appropriate with other data types.

    Transformation of quaternions is also supported:

    Normalised unit quaternion: >> q_u = q.unit()
    Conjugate of q:             >> q_c = q.conjugate()
    Reciprocal of q:            >> q_r = q.reciprocal() 

    Tolerance can be applied to the unit quaternion function to determine the
    unit quaternion to a desired accuracy.

    >> q_u = q.unit(tolerance=0.0001)

    """

    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        self.magnitude = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def __repr__(self):             #Appropriate representation
        return "Quaternion(%r + %r i + %r j + %r k)" % (self.w, self.x, self.y, self.z)

    def unit(self, tolerance=0):#Creates the equivalent unit quaternion
    #By default, produces unit quaternion with an arbitrary tolerance ie magnitude will not perfectly be 1 - but should be sufficiently close
    #if tolerance is given, recursively divides by magnitude until the new magnitude is sufficiently close to 1 as desired
        if not tolerance:
            return Quaternion(self.w/self.magnitude,
                              self.x/self.magnitude,
                              self.y/self.magnitude,
                              self.z/self.magnitude)
        else:
            new_q = self.unit()     #scarily close to recursion
            while abs(1 - new_q.magnitude) > tolerance:
                new_q = new_q.unit()
            return new_q

    def conjugate(self):            #Create the quaternion conjugate
        return Quaternion(self.w,
                          -self.x,
                          -self.y,
                          -self.z)

    def reciprocal(self):           #Create the reciprocal
        return (self.conjugate()/(self.magnitude**2))
    
    def __add__(self,other):        #Addition of quaternions
        if isinstance(other, Quaternion):
            return Quaternion(self.w+other.w,self.x+other.x, self.y+other.y, self.z+other.z)
        else:
            raise TypeError("Unable to broadcast together types Quaternion and %r." %type(other))

    def __sub__(self,other):        #Subtraction of quaternions
        if isinstance(other, Quaternion):
            return Quaternion(self.w-other.w,self.x-other.x, self.y-other.y, self.z-other.z)
        else:
            raise TypeError("Unable to broadcast together types Quaternion and %r." %type(other))

    def __mul__(self, other):       #Allow for right multiplication by scalars, matrices and quaternions
        if isinstance(other, Quaternion):
            x = self.x * other.w + self.y * other.z - self.z * other.y + self.w * other.x
            y = -self.x *other.z + self.y * other.w + self.z * other.x + self.w * other.y
            z = self.x * other.y - self.y * other.x + self.z * other.w + self.w * other.z
            w = -self.x *other.x - self.y * other.y - self.z * other.z + self.w * other.w
            return Quaternion(w,x,y,z)
        elif isinstance(other, np.ndarray):
            if other.shape[0] == 4: 
                return np.array([self.w,self.x,self.y,self.z]).dot(other)
            else:
                raise ValueError("Operand with shape (%r,%r) could not be broadcast with a quaternion." % other.shape)
        elif isinstance(other, (int, float)):
            return Quaternion(self.w*other,
                              self.x*other,
                              self.y*other,
                              self.z*other)
        else:
            raise TypeError("Multiplication of quaternion with given types is not supported.")

    def __rmul__(self,other):       #Allow for left multiplication by matrices and scalars (quaternion right multiplication handled above)
        if isinstance(other, np.ndarray):
            if other.shape[1] == 4: 
                return other.dot(np.array([self.w,self.x,self.y,self.z]))
            else:
                raise ValueError("Operand with shape (%r,%r) could not be broadcast with a quaternion." % other.shape)
        elif isinstance(other, (int, float)):
            return Quaternion(self.w*other,
                              self.x*other,
                              self.y*other,
                              self.z*other)
        else:
            raise TypeError("Multiplication of quaternion with given types is not supported.")

    def __truediv__(self, other):   #Allow for division by scalars and quaternions: division by other types is undefined
        if isinstance(other, Quaternion):   
            mag_2 = other.magnitude**2
            w = (other.w*self.w + other.x*self.x + other.y*self.y + other.z*self.z)/mag_2
            x = (other.w*self.x - other.x*self.w - other.y*self.z + other.z*self.y)/mag_2
            y = (other.w*self.y + other.x*self.z - other.y*self.w - other.z*self.x)/mag_2
            z = (other.w*self.z - other.x*self.y + other.y*self.x - other.z*self.w)/mag_2
            return Quaternion(w,x,y,z)
        elif isinstance(other, (int, float)):
            return Quaternion(self.w/other,
                              self.x/other,
                              self.y/other,
                              self.z/other)
        else:
            raise TypeError("Division of quaternion by given types is not supported.")
    
    __array_priority__ = 10000 #big number so numpy respects left matrix multiplication with quaternions
