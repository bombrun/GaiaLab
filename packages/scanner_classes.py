#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:59:19 2018

@author: vallevaro
"""

from quaternion import*
from scanner_functions import*
from sympy import*
from sympy import Line3D, Point3D
import numpy as np
import math
import numpy as np
              
class Observation:    
    def __init__(self, azimuth, altitude):
         self.azimuth = azimuth
         self.altitude = altitude
         self.coor = np.array([self.azimuth, self.altitude])
         
         self.x = np.cos(self.azimuth)*np.cos(self.altitude)
         self.y = np.sin(self.azimuth)*np.cos(self.altitude)
         self.z = np.sin(self.altitude)
         
         self.vector = unit_vector(np.array([self.x,self.y,self.z]))         
    
class Sky: 

    def __init__(self, n):
        self.elements = []
        
        for n in range(n):
            azimuth = np.random.uniform(0, (2*np.pi))
            altitude = np.random.uniform((-np.pi)/2, (np.pi)/2)
            obs = Observation(azimuth, altitude)
            self.elements.append(obs)  
    
class Satellite: 

    def __init__(self,z1,z2,z3, origin = Point3D(0.,0.,0.)): 
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
               
    def Scan(self, sky, zeta = np.radians(10.), stepphi = math.radians(1.), phi= math.radians(360.)):    
        
        self.indexes = []
        self.observations = [] 
        self.stars_zeta_angles = [] 
        self.times = [] 
        
        for idx, star in enumerate(sky.elements):    
            star_point = vector_to_point(star.vector)             
            star_line =  Line3D(self.xyplane.args[0], star_point)      #AL- both points in BCRS frame, so good.
            arc_angle_star_xyplane = self.xyplane.angle_between(star_line) #warning    #AL- normal vector to plane and star position vector both in BCRS frame, so should be good.
            if len(arc_angle_star_xyplane.args) == 2:
                zeta_angle_star_plane = float(arc_angle_star_xyplane.args[1])
            if len(arc_angle_star_xyplane.args) == 1:
                zeta_angle_star_plane = float(arc_angle_star_xyplane.args[0])
                
            self.stars_zeta_angles.append(zeta_angle_star_plane)  
            
            if  -zeta/2. < (zeta_angle_star_plane) < zeta/2.:        #AL- If star within zeta of plane, then continue.
                self.indexes.append(idx)
                proy_star_point = self.xyplane.projection(star_point) #warning    #AL- both plane and star position vector in BCRS frame, so proy_star_point in BCRS frame as well.
                print(idx)
                print(proy_star_point)
                proy_star_vector = point_to_vector(proy_star_point)   #AL- Fixed error here by changing method to use point.x etc rather than point[0]
                proy_star_vector_srs = SRS(self, proy_star_vector)
                #self.projections.append(proy_star_vector_srs)        #AL- Commented this line out, as projections list does not exist on the Satellite class at the moment... Isn´t used anywhere else yet so shouldn´t cause any problems (yet!)
                phi_angle_obs =  np.arctan2(proy_star_vector_srs[1], proy_star_vector_srs[0])
                
                if phi_angle_obs < 0.:
                    phi_angle_obs = phi_angle_obs + 2*np.pi
                observation = Observation(phi_angle_obs, zeta_angle_star_plane)
                self.observations.append(observation)
                
        for i in np.arange(0, phi, stepphi):
            self.ViewLine(i, 0)
            axis1phi = self.phi                 #maybe change this to +- stepphi/2 at some point? but careful that phi > 0
            axis2phi = self.phi + stepphi
            
            for observation in self.observations:
                if axis1phi < observation.azimuth and observation.azimuth < axis2phi:
                    self.times.append(i)

      
    
                                                                                                                                                                                                                                     

        
        
        
        
        
        
        
