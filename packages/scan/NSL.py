#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:59:19 2018

@author: vallevaro
"""

from plots import*
from functions import*

import numpy as np
import time


class Sky:
    def __init__(self, n):
        self.elements = []
        for i in range(n): #n number of objects in the time-interval given
            alpha = np.random.uniform(0,2*np.pi ) #time of possible observation in a day duration.
            delta = np.random.uniform(-np.pi/2, np.pi/2)
            self.elements.append(Source(alpha, delta))
            

class Source:
    def __init__(self, alpha, delta, mualpha = 0, mudelta = 0, parallax = 1):
        self.coor = xyz(alpha, delta) 

################################# SATELLITE ######################################
class Satellite: 

    def __init__(self,*args):  
        self.t = 0
        self.parameters(*args)
        self.reset()
        self.storinglist = []   
        
    def parameters(self, S = 4.035, epsilon = np.radians(23.26) , xi =  np.radians(55) , wz = 120):
        '''
        Args
        _______
        
        S: [float][constant] is equal to -> dz/dlambda. Equivalent to K constant (loops per year) whilst S is a velocity that describes the same as K effectively.
         
        epsilon: [float][constant] obliquity of equator
        
        xi: [float][constant] revolving angle
        
        wz: [float][constant] z component of spin phase, gives constant inertial rotation rate about zaxis in XYZ- frame (satellite frame).Provided in rad/s and transformed to rad/yr.
        
        '''
        #change of units
        wz = wz/206264.8062470946 #to radians/s
        wz = wz*60*60*24. #to radians/days
        
        #constant parameters
        self.S = S
        self.epsilon = epsilon
        self.xi = xi
        self.wz = wz
        
        #initial-value conditions for functions.
        self.ldot = (2*np.pi/365)
        self.t0 = 0.
        
        self.nu0 = 0.
        self.omega0 = 0.
        self.l0 = 0.
        self.beta0 = 0.

    def init_attitude(self):
        '''
        Called in __init__ method. 
        Initialises satellite.attitude according to initial values of all parameters. 
        '''
        q1 = Quaternion(np.cos(self.epsilon/2), np.sin(self.epsilon/2), 0, 0)                            #epsilon about l-axis
        q2 = Quaternion(np.cos(self.l0/2), 0,0,np.sin(self.l0/2))                                        #lambda about n-axis
        q3 = Quaternion(np.cos((self.nu0-np.pi/2.)/2), np.sin((self.nu0-np.pi/2.)/2), 0, 0)              #nu - 90 abbout l-axis
        q4 = Quaternion(np.cos((np.pi/2. - self.xi)/2), 0,np.sin((np.pi/2. - self.xi)/2), 0)             #90-xi about m-axis
        q5 = Quaternion(np.cos(self.omega0/2.), 0,0, np.sin(self.omega0/2.))                             #omega abbout n-axis (z-axis).
        
        q_total = q1*q2*q3*q4*q5
        return q_total
        
    def reset(self):
        '''
        Resets satellite to t = 0 and initialization status.
        '''
        #initialization of satellite.attitude.
        #self.storinglist = []
        self.attitude = self.init_attitude()
        
        #Initial orientation values set equal to initial-value conditions.
        self.t = self.t0
        self.nu = self.nu0
        self.omega = self.omega0
        self.lamb = self.l0
        self.beta = self.beta0
        
        #Initialization of z-axis and inertial rotation vector.
        z_quat = self.attitude * Quaternion(0,0,0,1) * self.attitude.conjugate()
        self.z_ = np.array([z_quat.x, z_quat.y, z_quat.z])
        self.w_ = np.cross(np.array([0,0,1]),self.z_)
        l_,j_,k_ = ljk(self.epsilon)
        self.s_ = l_*np.cos(self.lamb) + j_*np.sin(self.lamb)

    def update(self, dt):
        self.t = self.t + dt
        self.dL = self.ldot * dt
        self.lamb = self.lamb + self.dL
        
        #Updates Nu
        self.nudot = self.ldot*(np.sqrt(self.S**2 - np.cos(self.nu)**2) + np.cos(self.xi)*np.sin(self.nu))/np.sin(self.xi)
        self.dNu = self.nudot*dt
        self.nu = self.nu + self.dNu
        
        #LatitudeAngles
        self.lamb_z = self.lamb + np.arctan(np.tan(self.xi)*np.cos(self.nu))
        self.beta_z = np.arcsin(np.sin(self.xi)*np.sin(self.nu))
        
        #Updates Omega
        omegadot = self.wz - self.nudot*np.cos(self.xi) - self.ldot*np.sin(self.xi)*np.sin(self.nu)
        dOmega = omegadot * dt
        self.omega = self.omega + dOmega
        
        #calculates coordinates and then calculates s-vector 
        l_,j_,k_ = ljk(self.epsilon)
        self.s_ = l_*np.cos(self.lamb) + j_*np.sin(self.lamb)
        
        #Calculates z-axis from cross product: deltaz = (k x z)*ldot + (s x z)dNu
        zdot_ = np.cross(k_, self.z_)*self.ldot + np.cross(self.s_,self.z_)*self.nudot
        dz_ = zdot_*dt
        self.z_ = self.z_ + dz_
        self.z_ = self.z_ / np.linalg.linalg.norm(self.z_)       
        
        #updates inertial rotation vector
        self.w_ = k_*self.ldot + self.s_*self.nudot + self.z_*omegadot
        
        #calculates new attitude by deltaquat
        w_magnitude = np.sqrt((self.w_[0])**2 + (self.w_[0])**2 + (self.w_[0])**2)
        dzheta = w_magnitude * dt
        deltaquat = rotation_quaternion(self.w_, dzheta)
        self.attitude = deltaquat*self.attitude
        
        #x axis
        x_quat = self.attitude * Quaternion(0,1,0,0) * self.attitude.conjugate()
        self.x_ = quaternion_to_vector(x_quat)

    def move(self, ti, tf, dt):
        '''
        Moves following nls for tf - ti period of time at a dt spaced time.

        obj = (t, lambda, nu, omega, w_, s_, z_, attitude, x_)
        '''

        self.reset_to_time(ti)

        n_steps = (tf-ti)/dt
        self.t = ti
        for i in np.arange(n_steps):
            self.update(dt)
            self.storinglist.append([self.t, self.lamb, self.nu,
                                    self.omega, self.w_,self.s_, 
                                    self.z_, self.attitude, self.x_])
        self.storinglist.sort()  
    def reset_to_time(self, t):
        '''
        Resets all parameters of the satellite to input time t. 
        '''
        templist = [obj for obj in self.storinglist if obj[0] <= t]
        
        if len(templist) != 0:
            listelement  = templist[-1]
            
            self.t = listelement[0]
            self.lamb = listelement[1]
            self.nu = listelement[2]
            self.omega = listelement[3]
            self.w_ = listelement[4]
            self.s_ = listelement[5]
            self.z_ = listelement[6]
            self.attitude = listelement[7]
            self.x_ = listelement[8]
            
        else: 
            self.reset()
   
        #deltat = t%self.t
        #self.update(deltat)
      
    def get_attitude(self, t):
        self.reset_to_time(t)
        return self.attitude
        
    def get_xaxis(self, t):
        self.reset_to_time(t)
        return self.x_
       

 
class Scanner:
    '''
    __init__
    
    tolerance: [float][radians] for searching a star position with scanner axis.

    '''
    def __init__(self, ccd = 0.15, delta_z = 0.1, delta_y = 0.01, dAngle = np.radians(106.5)):
        self.dAngle = dAngle #not used, is for second ccd
        self.delta_z = delta_z
        self.delta_y = delta_y
        self.ccd = ccd
        
        self.obs_times = []
        self.starspositions = []
        self.times_to_scan_star = []
        
    def reset(self):
        self.obs_times = []
        self.starspositions = []
        self.times_to_scan_star = []
   
    def intercept(self, star, satellite):
        '''
        Calculates approx times where star transit occurs. 
        '''
        for obj in satellite.storinglist:
            t = obj[0]
            attitude = obj[7]
            
            starcoor_srs = SRS(attitude, star.coor)
            xyproy_star_srs = np.array([starcoor_srs[0], starcoor_srs[1], 0])
            xzproy_star_srs = np.array([starcoor_srs[0], 0, starcoor_srs[2]])
            
            x_telescope1 = obj[8]
            x_srs_telescope1 = SRS(attitude, x_telescope1)
            
            width_angle = 2*np.arctan2(self.ccd/2, x_srs_telescope1[0])
            aperture_angle = 2*np.arctan2(self.delta_z/2, x_srs_telescope1[0])
            
            if np.arccos(np.dot(xyproy_star_srs, x_srs_telescope1)) < width_angle:
                if np.arccos(np.dot(xzproy_star_srs, x_srs_telescope1)) < aperture_angle:
                    self.times_to_scan_star.append(t)

    def zoom_intercept(self, satellite,  dt = 0.001):
        for idx, t in enumerate(self.times_to_scan_star):
            while (idx+1) < len(self.times_to_scan_star):
                satellite.move(t, self.times_to_scan_star[idx + 1], dt)
            else:
                satellite.move(t, t+0.5, dt)


def StarFinder(satellite, scanner, sky): 
    scanner.reset()
    for star in sky.elements:

        scanner.intercept(star, satellite)
        scanner.zoom_intercept(satellite)

        for obj in satellite.storinglist:
            x_telescope1 = obj[8]  #wrt bcrs frame
            t  = obj[0]
            attitude = obj[7]
            
            #change frame to SRS for scanning.
            x_srs_telescope1= SRS(attitude, x_telescope1)     #approx should be (1,0,0) always.
            star_srs_coor = SRS(attitude, star.coor)
            
            #scanner parameters
            aperture_angle = np.arctan2(scanner.delta_z/2, x_srs_telescope1[0])
            
            #condition for xaxis:
            if  np.arccos(np.dot(star_srs_coor, x_srs_telescope1)) < aperture_angle:
                #condition for z axis
                if np.abs(star_srs_coor[2] - x_srs_telescope1[2]) < scanner.delta_z:
                    #condition for y axis:
                    if np.abs(star_srs_coor[1] - x_srs_telescope1[1]) < scanner.delta_y:
                        scanner.obs_times.append(t)
                        scanner.starspositions.append(BCRS(attitude, x_srs_telescope1))

###########################################################################
def Run():
    start_time = time.time()
    sky = Sky(1)
    scan = Scanner()
    gaia = Satellite()
    gaia.storinglist = []
    gaia.move(0,365, 0.01)
    
    StarFinder(gaia, scan, sky)

    seconds = time.time() - start_time
    m, s = divmod(seconds, 365)
    h, m = divmod(m, 60)
    
    print ("%d:%02d:%02d" % (h, m, s))
    print (len(scan.starspositions))
    return scan, gaia, sky

    


