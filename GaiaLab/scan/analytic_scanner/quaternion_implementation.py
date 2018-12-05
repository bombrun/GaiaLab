"""
Quaternion class implementation for python.

Offers native integration of operators between quaternions generated and
numpy data structures.

:Authors:
    Toby James 2018
    LucaZampieri 2018 (modifications)
"""

import numpy as np


class Quaternion():
    """
    Quaternion class implemented to allow manipulation of quaternions with
    each other, matrices (including, where appropriate, vectors) and scalars.

    Declare a quaternion of the form q = w + xi + yj +zk by

    >>> q = Quaternion(w,x,y,z)

    Addition, subtraction, multiplication and division are all supported natively
    between quaternions and where appropriate with other data types.

    Transformation of quaternions is also supported:

    Normalised unit quaternion:     >>> q_u = q.unit()
    Conjugate of q:                 >>> q_c = q.conjugate()
    Reciprocal or inverse of q:     >>> q_r = q.reciprocal() or >>> q_i = q.inverse(). They are equivalent.

    Tolerance can be applied to the unit quaternion function to determine the
    unit quaternion to a desired accuracy.

    >>> q_u = q.unit(tolerance=0.0001)

    """

    def __init__(self, w=None, x=None, y=None, z=None, vector=None, angle=None):

        if (vector is not None) and (angle is not None):
            w, x, y, z = self.parameters_from_vector_and_angle(vector, angle)
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        self.magnitude = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def __repr__(self):             # Appropriate representation
        return "Quaternion(%r + %r i + %r j + %r k)" % (self.w, self.x, self.y, self.z)

    def compute_magnitude(self):
        self.magnitude = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        return self.magnitude

    def unit(self, tolerance=0):  # Creates the equivalent unit quaternion
        """
        # By default, produces unit quaternion with an arbitrary tolerance ie magnitude
        # will not perfectly be 1 - but should be sufficiently close
        # If tolerance is given, recursively divides by magnitude until the new magnitude
        # is sufficiently close to 1 as desired
        """
        if not tolerance:
            return Quaternion(self.w/self.magnitude,
                              self.x/self.magnitude,
                              self.y/self.magnitude,
                              self.z/self.magnitude)
        else:
            new_q = self.unit()     # scarily close to recursion
            while abs(1 - new_q.magnitude) > tolerance:
                new_q = new_q.unit()
            return new_q

    def reset_with_matrix(self, m):
        raise ValueError('this function is incorrectly implemented')
        if m[2, 2] < 0:
            if m[0, 0] > m[1, 1]:
                trace = 1 + m[0, 0] - m[1, 1] - m[2, 2]
                x = m[0, 1]+m[1, 0]
                y = m[2, 0]+m[0, 2]
                z = m[1, 2]-m[2, 1]
                quat = Quaternion(x, y, z, trace)
            else:
                trace = 1 - m[0, 0] + m[1, 1] - m[2, 2]
                quat = Quaternion(trace, m[1, 2]+m[2, 1], m[2, 0]-m[0, 2], m[0, 1]+m[1, 0])
        else:
            if m[0, 0] < -m[1, 1]:
                trace = 1 - m[0, 0] - m[1, 1] + m[2, 2]
                x = m[2, 0]+m[0, 2]
                y = m[1, 2]+m[2, 1]
                z = m[0, 1]-m[1, 0]
                quat = Quaternion(y, trace, z, x)
            else:
                trace = 1 + m[0, 0] + m[1, 1] + m[2, 2]
                x = m[1, 2]-m[2, 1]
                y = m[2, 0]-m[0, 2]
                z = m[0, 1]-m[1, 0]
                quat = Quaternion(y, z, trace, x)
        quat = quat*0.5/np.sqrt(trace)
        return quat

    def parameters_from_vector_and_angle(self, vector, angle):
        """
        Quaternion parameters equivalent to rotation about (vector) by an (angle).

        :param vector:  [np.array] axis of rotation
        :param angle: [deg] amplitude of rotation (and directions)
        :returns: (t, x, y, z) corresponding quaternion parameters
        """
        vector = vector / np.linalg.norm(vector)
        t = np.cos(angle/2.)
        x = np.sin(angle/2.) * vector[0]
        y = np.sin(angle/2.) * vector[1]
        z = np.sin(angle/2.) * vector[2]
        return (t, x, y, z)

    def conjugate(self):            # Create the quaternion conjugate
        return Quaternion(self.w,
                          -self.x,
                          -self.y,
                          -self.z)

    def reciprocal(self):           # Create the reciprocal
        return (self.conjugate()/(self.magnitude**2))

    def inverse(self):              # Same as reciprocal
        """
        Computes:

        .. math::
            q^{-1} = \\frac{w - xi - yj - zk}{w^2 + x^2 + y^2 + z^2}
        """
        return self.reciprocal()
        # return self.conjugate()

    def __add__(self, other):        # Addition of quaternions
        if isinstance(other, Quaternion):
            return Quaternion(self.w+other.w, self.x+other.x, self.y+other.y, self.z+other.z)
        else:
            raise TypeError("Unable to broadcast together types Quaternion and %r." % type(other))

    def __sub__(self, other):        # Subtraction of quaternions
        if isinstance(other, Quaternion):
            return Quaternion(self.w-other.w, self.x-other.x, self.y-other.y, self.z-other.z)
        else:
            raise TypeError("Unable to broadcast together types Quaternion and %r." % type(other))

    # For multiplication, python 3.5+ supports @ as the matrix multiplication operator.
    # Improvements in readability may be gained by replacing .dot functions with @.

    # It is not (currently) in __future__, so from __future__ import does not allow for
    # use with python versions < 3.5.

    # No non-stylistic improvements, it breaks backwards compatibility and more than 2
    # matrices are never multiplied here therefore it is probably not currently worth
    # implementing.

    def __mul__(self, other):
        # Allow for right multiplication by scalars, matrices and quaternions
        if isinstance(other, Quaternion):
            x = self.x * other.w + self.y * other.z - self.z * other.y + self.w * other.x
            y = -self.x * other.z + self.y * other.w + self.z * other.x + self.w * other.y
            z = self.x * other.y - self.y * other.x + self.z * other.w + self.w * other.z
            w = -self.x * other.x - self.y * other.y - self.z * other.z + self.w * other.w
            return Quaternion(w, x, y, z)
        elif isinstance(other, np.ndarray):
            if other.shape[0] == 4:
                return np.array([self.w, self.x, self.y, self.z]).dot(other)
            else:
                raise ValueError("Operand with shape (%r,%r) could not be broadcast with a quaternion." % other.shape)
        elif isinstance(other, (int, float)):
            return Quaternion(self.w*other,
                              self.x*other,
                              self.y*other,
                              self.z*other)
        else:
            raise TypeError("Multiplication of quaternion with %r is not supported." % type(other))

    def __rmul__(self, other):
        # Allow for left multiplication by matrices and scalars (quaternion right multiplication handled above)
        if isinstance(other, np.ndarray):
            if other.shape[1] == 4:
                return other.dot(np.array([self.w, self.x, self.y, self.z]))
            else:
                raise ValueError("Operand with shape (%r,%r) could not be broadcast with a quaternion." % other.shape)
        elif isinstance(other, (int, float)):
            return Quaternion(self.w*other,
                              self.x*other,
                              self.y*other,
                              self.z*other)
        else:
            raise TypeError("Multiplication of quaternion with %r is not supported." % type(other))

    def __truediv__(self, other):
        # Allow for division by scalars and quaternions: division by other types is undefined
        if isinstance(other, Quaternion):
            mag_2 = other.magnitude**2
            w = (other.w*self.w + other.x*self.x + other.y*self.y + other.z*self.z)/mag_2
            x = (other.w*self.x - other.x*self.w - other.y*self.z + other.z*self.y)/mag_2
            y = (other.w*self.y + other.x*self.z - other.y*self.w - other.z*self.x)/mag_2
            z = (other.w*self.z - other.x*self.y + other.y*self.x - other.z*self.w)/mag_2
            return Quaternion(w, x, y, z)
        elif isinstance(other, (int, float)):
            return Quaternion(self.w/other,
                              self.x/other,
                              self.y/other,
                              self.z/other)
        else:

            raise TypeError("Division of quaternion by given types is not supported.")

    def to_vector(self):
        return np.array([self.x, self.y, self.z])

    def to_4D_vector(self):
        return np.array([self.w, self.x, self.y, self.z])

    def basis(self):
        """
        Rotation matrix associated with quaternion.
        See https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        """
        s = self.magnitude
        a_11 = 1 - 2*s*((self.y)**2 + (self.z)**2)
        a_12 = 2*s*(self.x*self.y - self.z*self.w)
        a_13 = 2*s*(self.x*self.z + self.y*self.w)
        a_21 = 2*s*(self.x*self.y + self.z*self.w)
        a_22 = 1 - 2*s*((self.x)**2 + (self.z)**2)
        a_23 = 2*s*(self.y*self.z - self.x*self.w)
        a_31 = 2*s*(self.x*self.z - self.y*self.w)
        a_32 = 2*s*(self.y*self.z + self.x*self.w)
        a_33 = 1 - 2*s*((self.x)**2 + (self.y)**2)

        A = np.array([[a_11, a_12, a_13],
                      [a_21, a_22, a_23],
                      [a_31, a_32, a_33]])

        return A  # A.reshape(3, 3)

    def rotation_axis_and_angle(self, tol=1e-10):
        R = self.unit().basis()
        if np.allclose(R, R.T, atol=tol):
            raise ValueError('Method not implemented for near symmetric matrices!')
        axis = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos((np.trace(R) - 1) / 2)
        return axis, angle

    __array_priority__ = 10000  # big number so numpy respects left matrix multiplication with quaternions
