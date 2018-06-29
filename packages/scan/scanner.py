#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:59:19 2018

@author: vallevaro
"""

from quaternion import*
from sympy import*
from sympy import Line3D, Point3D
import numpy as np
import math
import matplotlib.pyplot as plt

class Observation:    
    '''
    Creates and object equivalent to an observation. 

    Horizontal Celestial Coordinates: self.coor = (azimuth, altitude)
    
    Calculates equivalence to Cartesian Coordinates: self.vector = (x, y, z)
    '''
    def __init__(self, azimuth, altitude):
         self.azimuth = azimuth
         self.altitude = altitude
         self.coor = np.array([self.azimuth, self.altitude])
         
         self.x = np.cos(self.azimuth)*np.cos(self.altitude)
         self.y = np.sin(self.azimuth)*np.cos(self.altitude)
         self.z = np.sin(self.altitude)
         
         self.observations = []  
         self.measurements = []  
         self.times = [] 
         self.indexes = []
            
         self.vector = unit_vector(np.array([self.x,self.y,self.z]))   
                          
class Sky:  
    '''
    Creates a sky in the unit sphere, in the BCRS frame.
    List of elements: sky.elements
    '''
    def __init__(self, n):
        self.elements = []
        
        for n in range(n):
            azimuth = np.random.uniform(0, (2*np.pi))
            altitude = np.random.uniform(-np.pi/2., np.pi/2)
            obs = Observation(azimuth, altitude)
            self.elements.append(obs)  
    
class Satellite: 
    '''
    Satellite object is resumed to be a plane.
    A plane is determined completely by a Point and a Vector.
    
    __init__: define the satellite by given a point and a vector - default: point at (0,0,0)
    
            satellite.Rotate(_): introduce in _ a quaternion (from Quaternion class) to rotate the plane (aka satellite)
    
            satellite.Scan(_): introduce in _ sky to be scanned. 
            satellite.observations = objects with coordinates in the SRS frame.
            satellite.measurements = objects with coordinates in the BCRS frame.
            satellite.times = angle at which a star interception occurs.
            satellite.indexes = the index of the observations made wrt the sky catalogue. 
    Scan():
        
        Azimuth angle (Phi):indicates the radians swept by the scanner in the satellite plane. 
        This scanner checks stars by increasing a 'deltaphi' radians in the satellite plane.
        Altitude angle (zeta):the altitude width angle of the scanner (width of vertical field of view wrt satellite plane)

    '''
    def __init__(self,z1,z2,z3, origin = Point3D(0,0,0)):  
        self.zaxis = unit_vector(np.array([z1,z2,z3]))               
        self.xyplane = Plane(origin, vector_to_point(self.zaxis))
        self.attitude = Quaternion(1.,0.,0.,0.).unit()

   
    def Rotate(self, newrotation):        
        self.attitude = newrotation.unit() * self.attitude 
        self.attitude.basis()
                                          
        self.zaxis = unit_vector(np.dot(self.attitude.A, self.zaxis))      
        self.xyplane = Plane((0.,0.,0.), vector_to_point(self.zaxis))
        
    def ViewLine(self, phi, zeta):
        self.phi = phi
        self.zeta = zeta    

    def Scan(self, sky, zeta = np.radians(5.), deltaphi = np.radians(1.), time = 6, deltatime = 1/60., omega = np.pi/3):   

        '''
        Calculates in the BCRS the angle between the plane of the satellite and the line from the centre of the satellite to the star.
        This angle is - zeta_angle_star_plane.
        '''
        self.observations = []  
        self.measurements = []  
        self.times = [] 
        self.indexes = []
        
        for idx, star in enumerate(sky.elements):    
            star_point = vector_to_point(star.vector)             
            star_line =  Line3D(self.xyplane.args[0], star_point)      
            arc_angle_star_xyplane = self.xyplane.angle_between(star_line)    
            if len(arc_angle_star_xyplane.args) == 2:
                zeta_angle_star_plane = -float(arc_angle_star_xyplane.args[1])
            if len(arc_angle_star_xyplane.args) == 1:
                zeta_angle_star_plane = float(arc_angle_star_xyplane.args[0])
                 

            if  -zeta/2. < (zeta_angle_star_plane) < zeta/2.:       
                self.indexes.append(idx)
                
                proy_star_point = self.xyplane.projection(star_point)             
                proy_star_vector = point_to_vector(proy_star_point)                
                proy_star_vector_srs = SRS(self, proy_star_vector) 
                               
                phi_angle_obs =  np.arctan2(float(proy_star_vector_srs[1]), float(proy_star_vector_srs[0]))
                zeta_angle = np.arctan2(float(proy_star_vector_srs[2]), float(np.sqrt((proy_star_vector_srs[0])**2+(proy_star_vector_srs[0])**2)))
                
                if phi_angle_obs < 0.:
                    phi_angle_obs = phi_angle_obs + 2*np.pi
                observation = Observation(phi_angle_obs, zeta_angle)
                self.observations.append(observation)
        
        '''
        Once observations are made, now we pass the scan to see at what times if the star in the detector's range
        '''
        #maybe change this to +- deltaphiphi/2 at some point? but careful that phi > 0
        for t in np.arange(0, time, deltatime):
            angle = omega * t
            deltaangle = omega * deltatime
            self.ViewLine(angle, 0)
            axis1phi = self.phi%(2*np.pi)            
            axis2phi = (self.phi + deltaangle)%(2*np.pi)
            
            for observation in self.observations:
                if axis1phi < observation.azimuth and observation.azimuth < axis2phi:
                    self.times.append(t)  
                                                                                                                                                                                                                                  
        
################################## FUNCTIONS ##################################

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

    
def SRS(satellite, vector):
    '''
    Changes coordinates of a vector in BCRS to SRS frame.
    '''
    q_vector_bcrs= vector_to_quaternion(vector)
    q_vector_srs = satellite.attitude * q_vector_bcrs * satellite.attitude.conjugate()  #AL- Swapped to put conjugate on RHS of vector
    return np.array([q_vector_srs.x, q_vector_srs.y, q_vector_srs.z])    
   
def BCRS(satellite, vector):
    '''
    Changes coordinates of a vector in SRS to BCRS frame.
    '''
    q_vector_srs= vector_to_quaternion(vector)
    #q_vector_bcrs = satellite.attitude * q_vector_srs * satellite.attitude.conjugate()
    q_vector_bcrs = satellite.attitude.conjugate() * q_vector_srs * satellite.attitude  #AL- Swapped to put conjugate on LHS of vector
    
    return np.array([q_vector_bcrs.x, q_vector_bcrs.y, q_vector_bcrs.z])
    
def Measurements(satellite): 
    '''
    Takes all observation objects of the satellite (which are in the SRS frame) and converts them into the BCRS frame, making them observation-objects.
    self.measurements are objects with bcrs coordinates.
    '''
    satellite.measurements =[] 
    for obs in satellite.observations: 
        star_vector = BCRS(satellite, obs.vector)
        alpha = np.arctan2(star_vector[1], star_vector[0])
        delta = np.arctan2(star_vector[2], np.sqrt(star_vector[0]**2 + star_vector[1]**2))
        if alpha < 0 :
            alpha = alpha + 2*np.pi
        star = Observation(alpha, delta)
        satellite.measurements.append(star) 
      
def Psi(satellite, sky):
    '''
    Calculates the difference between the coordinates of a star versus its correspondient coordinates (bcrs-framed) from Gaia.
    '''
    bcrs_stars_vector = [BCRS(satellite, obs.vector) for obs in satellite.observations]
    list_true_star_vector = [sky.elements[idx].vector for idx in satellite.indexes]
    diff = np.subtract(bcrs_stars_vector, list_true_star_vector)
    return    diff

    
def Plot(satellite, sky):   
    '''
    Plot: measurements (coordinates of stars measured by gaia and transformed into BCRS frame) vs true coordinates of the detected stars. 
    '''
    Measurements(satellite)
    azimuth_obs = [star.coor[0] for star in satellite.measurements]
    altitude_obs = [star.coor[1] for star in satellite.measurements]
    
    azimuth_star = [sky.elements[idx].coor[0] for idx in satellite.indexes]
    altitude_star = [sky.elements[idx].coor[1] for idx in satellite.indexes]
    
    plt.figure()   
    plt.grid()
    plt.ylabel('Altitude (rad)')
    plt.xlabel('Azimuth (rad)')
    plt.title('Measurements vs True Stars')
    
    red_dot, = plt.plot(azimuth_obs, altitude_obs, 'r*')
    blue_dot, = plt.plot(azimuth_star, altitude_star, 'bo')

    plt.legend([red_dot, (red_dot, blue_dot)], ["Obs", "True Star"])
    plt.show()
    

    
    

    
        
        
        
        
        
        
