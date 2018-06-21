import numpy as np 
from sympy import*
from sympy import Line3D, Point3D
from quaternion import*
from scanner_classes import*

def vector(x,y,z):
    return np.array([x,y,z])

def unit_vector(vector): 
    return vector / np.linalg.norm(vector) 
            
def vector_to_point(vector):
    return Point3D(vector[0], vector[1], vector[2])
    
def point_to_vector(point):
    return np.array(point[0], point[1], point[2])           

def vector_to_quaternion(vector):
    return Quaternion(0, vector[0], vector[1], vector[2])
       
def rotation_quaternion(vector, angle):     
    #Calculates Quaternion equivalent to a rotation given by a vector and a angle in radians.
    vector = unit_vector(vector)   
    t = np.cos(angle/2.)
    x = np.sin(angle/2.)*vector[0]
    y = np.sin(angle/2.)*vector[1]
    z = np.sin(angle/2.)*vector[2]
    
    qvector = Quaternion(t,x,y,z)
    return qvector

def Psi(satellite, sky):

    bcrs_stars_vector = [bcrs(satellite, star) for star in satellite.observations]
    list_true_star_vector = [sky.elements[idx].vector for idx in satellite.indexes]
    diff = np.subtract(bcrs_stars_vector, list_true_star_vector)
    
    return bcrs_stars_vector, list_true_star_vector, diff
    
def bcrs(satellite, star):
    starvector_srs = star.vector
    q_star_vector_srs= vector_to_quaternion(starvector_srs)
    q_star_vector_bcrs = satellite.attitude * q_star_vector_srs * satellite.attitude.conjugate()
    return np.array([q_star_vector_bcrs.x, q_star_vector_bcrs.y, q_star_vector_bcrs.z])
    
    
def _bcrs(satellite, sky):
    bcrs_stars_vector = [] 
    satellite.attitude.basis()
    A = satellite.attitude.A
    invA = np.linalg.inv(A)
    true_star_bcrs = [sky.elements[idx].vector for idx in satellite.indexes]
    for star in satellite.observations:        
        star_bcrs_vector = np.dot(invA, star.vector)
        bcrs_stars_vector.append(star_bcrs_vector)
    return bcrs_stars_vector, true_star_bcrs
    

    
    