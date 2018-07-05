#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:59:19 2018

@author: vallevaro
"""

from quaternion import*
from sympy import*
from frameRotation import*
from functions import*

from sympy import*
from sympy import Line3D, Point3D
from sympy import Symbol

import numpy as np
import math
import matplotlib.pyplot as plt
import time
import datetime

    
    
class Star: 
    def __init__(self, source):
        azimuth = source.param[0]
        altitude = source.param[1]
        srs_star_vector = xyz(azimuth, altitude)
        self.coorsrs = srs_star_vector
        
class Source:
    """    
    source detected by satellite, all parameters with respect to SRS frame.
    
    Args
    ----------
    
    alpha : [float] azimuth angle a.k.a. right ascension.
    delta: [float] algular distance from celestial equator a.k.a. declination.
    omega: [float] annual parallax, the apaprent shift in angular position when viewed from opposite sides of the sun.
    mualphastar: [float] mualpha*cos[delta] proper motion in alpha
    mudelta: [float] proper motion in delta.
    muomega: [float] radial proper motion, motion in the line of sight direction.
             
    Attributes
    -------
    .parameters: [array] shape (alpha, delta, omega, mualphastar, mudelta, muomega)
    
    """
    def __init__(self, alpha, delta, omega, mualphastar, mudelta, muomega):
        self.param = np.array([alpha, delta, omega, mualpha, mudelta, muomega])

def Observation(satellite, t):
    return None
    
        
class Satellite: 

    def __init__(self, *args):  
        self.Parameters(*args)
        
    def Parameters(self, S = 4.035, epsilon = np.radians(23.26) , xi =  np.radians(55) , wz = 0.0005817764173):
        '''
        Args
        _______
        
        S: [float][constant] is equal to -> dz/dlambda. Equivalent to K constant (loops per year) whilst S is a velocity that describes the same as K effectively.
         
        epsilon: [float][constant] obliquity of equator
        
        xi: [float][constant] revolving angle
        
        omegaz: [float][constant] z component of spin phase, gives constant inertial rotation rate about zaxis in XYZ- frame (satellite frame). 
        
        '''
        #constant parameters
        self.S = S
        self.epsilon = epsilon
        self.xi = xi
        self.wz = wz
        
        #initial-value conditions for functions.
        self.ldot = (2*np.pi/365)
        
        self.nu0 = 0.5
        self.omega0 = 0.4
        self.l0 = 0.12
        
        #initialization of satellite.attitude.
        self.attitude = self.InitAttitude()
        
        #Initial orientation values set equal to initial-value conditions.
        self.nu = self.nu0
        self.omega = self.omega0
        self.l = self.l0
        
        
        
        z_quat = self.attitude * Quaternion(0,0,0,1) * self.attitude.conjugate()
        self.z_ = [z_quat.x, z_quat.y, z_quat.z]
        self.w_ = np.cross([0,0,1],self.z_)
        
    def InitAttitude(self):
        '''
        Called in __init__ method. 
        Initialises satellite.attitude according to initial values of all parameters. 
        '''
        q1 = Quaternion(np.sin(self.epsilon/2), 0, 0, np.cos(self.epsilon/2))                            #epsilon about l-axis
        q2 = Quaternion(0,0,np.sin(self.l0/2),np.cos(self.l0/2))                                         #lanbda about n-axis
        q3 = Quaternion(np.sin((self.nu0-np.pi/2.)/2), 0, 0, np.cos((self.nu0-np.pi/2.)/2))              #nu - 90 abbout l-axis
        q4 = Quaternion(0,np.sin((np.pi/2. - self.xi)/2), 0, np.cos((np.pi/2. - self.xi)/2))             #90-xi about m-axis
        q5 = Quaternion(0,0, np.sin(self.omega0/2.), np.cos(self.omega0/2.))                             #omega abbout n-axis (z-axis).
        
        q_total = q1*q2*q3*q4*q5
        return q_total
        
    def Attitude(self):

        #calculate w vector
        #calculate z from w vector
        #calculate attitude from w. 
        return None
            
    def WMatrix(self, w_):
        w_[0] = wl
        w_[1] = wm
        w_[2] = wn
        
        
        wMatrix = np.array([0, -wn, +wm, wl,+wn, 0, -wl, +wm, -wm, +wl, 0, wn, -wl, -wm, -wn, 0])   #need to change this to have proper structure of quaternion (x,y,z,t) rather than (t,x,y,z)
        wMatrix = wMatrix.reshape(4,4)
       
        return wMatrix

    def Update(self, dt):
        self.t = self.t + dt
        dL = self.ldot * dt
        self.l = self.l + dL
        
        #Updates Nu
        nudot = (self.ldot + (np.sqrt(self.S**2- np.cos(self.nu)**2) + np.cos(self.xi)*np.sin(self.nu))/np.sin(self.xi)) * self.ldot  # = dNu/dL * dL/dt
        dNu = nudot*dt
        self.nu = self.nu + dNu
        
        #Updates Omega
        omegadot = self.wz - nudot*np.cos(self.xi) - self.ldot*np.sin(self.xi)*np.sin(self.nu)
        dOmega = omegadot * dt
        self.omega = self.omega + dOmega
        
        #calculates coordinates and then calculates s-vector 
        l_,j_,k_ = ljk(self.epsilon)
        self.s_ = l_*np.cos(self.l) + j_*np.sin(self.l)
        
        #Calculates z-axis from cross product: deltaz = (k x z)*ldot + (s x z)dNu
        zdot_ = np.cross(k_, self.z_)*self.ldot + np.cross(self.s_,self.z_)*nudot
        dz_ = zdot_*dt
        self.z_ = self.z_ + dz_
        
        self.w_ = k_*self.ldot + self.s_*nudot + self.z_*omegadot
        
        wMatrix = WMatrix(self, self.w_)
                
        qdot = 0.5*wMatrix * self.attitude
        dq = qdot * dt
        #check this and see what matrix * quaternion gives and what dq and qdot give
        #self.attitude = dq*self.attidute 

        
        

    
        
        
        
        
        
        
