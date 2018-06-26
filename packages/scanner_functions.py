import numpy as np 
import matplotlib as plt
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
    #return np.array([point[0], point[1], point[2]])           
    return np.array([point.x, point.y, point.z])           

def vector_to_quaternion(vector):
    return Quaternion(0, float(vector[0]), float(vector[1]), float(vector[2]))  #added float arguments to prevent Point3D fractions from being passed to the quaternion class.
       
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

    bcrs_stars_vector = [BCRS(satellite, obs) for obs in satellite.observations]     #matrix and quaternion give same result, good. 
    list_true_star_vector = [sky.elements[idx].vector for idx in satellite.indexes]
    diff = np.subtract(bcrs_stars_vector, list_true_star_vector)
    
    return diff

    
def SRS(satellite, vector):
    q_vector_bcrs= vector_to_quaternion(vector)
    q_vector_srs = satellite.attitude.conjugate() * q_vector_bcrs * satellite.attitude
    return np.array([q_vector_srs.x, q_vector_srs.y, q_vector_srs.z])    
    

    
def BCRS(satellite, star):
    starvector_srs = star.vector
    q_star_vector_srs= vector_to_quaternion(starvector_srs)
    q_star_vector_bcrs = satellite.attitude * q_star_vector_srs * satellite.attitude.conjugate()
    return np.array([q_star_vector_bcrs.x, q_star_vector_bcrs.y, q_star_vector_bcrs.z])
    
def CreateStars(satellite): #there is differences for signs, bad.
    satellite.measurements = [] #this is in bcrs frame
    for obs in satellite.observations: 
        star_vector = BCRS(satellite, obs)
        star = Star(star_vector[0], star_vector[1], star_vector[2])
        satellite.measurements.append(star) 
        
def Plot(satellite, sky):   
    CreateStars(satellite) 
    azimuth_obs = [star.coor[0] for star in satellite.measurements]
    altitude_obs = [star.coor[1] for star in satellite.measurements]
    
    azimuth_star = [sky.elements[idx].coor[0] for idx in satellite.indexes]
    altitude_star = [sky.elements[idx].coor[1] for idx in satellite.indexes]
    
    plt.close()
    plt.figure()   
    plt.grid()
    plt.ylabel('Altitude (rad)')
    plt.xlabel('Azimuth (rad)')
    plt.title('Measurements vs True Stars')
    
    red_dot, = plt.plot(azimuth_obs, altitude_obs, 'r*')
    blue_dot, = plt.plot(azimuth_star, altitude_star, 'bo')

    plt.legend([red_dot, (red_dot, blue_dot)], ["Obs", "True Star"])
    plt.show()

    
    