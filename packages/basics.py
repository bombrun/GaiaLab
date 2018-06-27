from sympy import*
from sympy import Line3D, Point3D
import numpy as np
import math
import matplotlib.pyplot as plt
def vector(x,y,z): 
    return np.array([x,y,z])

def unit_vector(vector): 
    return vector / np.linalg.norm(vector) 
            
def vector_to_point(vector):
    return Point3D(vector[0], vector[1], vector[2])
    
def point_to_vector(point):
    #return np.array([point[0], point[1], point[2]])           
    return np.array([point.x, point.y, point.z])           

def vector_to_quaternion(vector):
    return Quaternion(0, float(vector[0]), float(vector[1]), float(vector[2]))  #added float arguments to prevent Point3D fractions from being passed to the quaternion class.
       
def rotation_quaternion(vector, angle): 
    '''    
    Calculates Quaternion equivalent to a rotation given by a vector and a angle in radians.
    '''
    vector = unit_vector(vector)   
    t = np.cos(angle/2.)
    x = np.sin(angle/2.)*vector[0]
    y = np.sin(angle/2.)*vector[1]
    z = np.sin(angle/2.)*vector[2]
    
    qvector = Quaternion(t,x,y,z)
    return qvector