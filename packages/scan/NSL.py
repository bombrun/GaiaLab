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
import time
import datetime

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
    
    
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

    def __init__(self,*args):  
        self.t = 0
        self.Parameters(*args)
        self.Reset()       
    def Parameters(self, S = 4.035, epsilon = np.radians(23.26) , xi =  np.radians(55) , wz = 120):
        '''
        Args
        _______
        
        S: [float][constant] is equal to -> dz/dlambda. Equivalent to K constant (loops per year) whilst S is a velocity that describes the same as K effectively.
         
        epsilon: [float][constant] obliquity of equator
        
        xi: [float][constant] revolving angle
        
        omegaz: [float][constant] z component of spin phase, gives constant inertial rotation rate about zaxis in XYZ- frame (satellite frame).Provided in rad/s and transformed to rad/yr.
        
        '''
        #change of units
        wz = wz/206264.8062470946 #to radians/s
        wz = wz*60*60*24. #to radians/days
        
        #constant parameters
        self.S = S
        self.epsilon = epsilon
        self.xi = xi
        self.wz = wz

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
        
    def Reset(self):
        '''
        Resets satellite to t = 0 and initialization status.
        '''
        #initial-value conditions for functions.
        self.ldot = (2*np.pi/365)
        self.t = 0.
        
        self.nu0 = 0.
        self.omega0 = 0.
        self.l0 = 0.
        self.beta0 = 0.
        
        #initialization of satellite.attitude.
        self.attitude = self.InitAttitude()
        
        #Initial orientation values set equal to initial-value conditions.
        self.nu = self.nu0
        self.omega = self.omega0
        self.l = self.l0
        self.beta = self.beta0
        
        #Initialization of z-axis and inertial rotation vector.
        z_quat = self.attitude * Quaternion(0,0,0,1) * self.attitude.conjugate()
        self.z_ = [z_quat.x, z_quat.y, z_quat.z]
        self.w_ = np.cross([0,0,1],self.z_)

    def Update(self, dt):
        self.t = self.t + dt
        dL = self.ldot * dt
        self.l = self.l + dL
        
        #Updates Nu
        nudot = (self.ldot*(np.sqrt(self.S**2- np.cos(self.nu)**2) + np.cos(self.xi)*np.sin(self.nu))/np.sin(self.xi)) * self.ldot  # = dNu/dL * dL/dt
        dNu = nudot*dt
        self.nu = self.nu + dNu
        
        #LatitudeAngles
        self.lamb = self.l + np.arctan(np.tan(self.xi)*np.cos(self.nu))
        self.beta = np.arcsin(np.sin(self.xi)*np.sin(self.nu))
        
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
        #AT THE MOMENT Z AND W ARE THE SAME. 
        #updates inertial rotation vector
        self.w_ = k_*self.ldot + self.s_*nudot + self.z_*omegadot
        
        #calculates new attitude by deltaquat
        wquat = vector_to_quaternion(self.w_)
        dzheta = wquat.magnitude * dt
        deltaquat = rotation_quaternion(self.w_, dzheta)
        self.attitude = deltaquat*self.attitude
        
    def GenerateLists(self, dt, n):
        self.w_list = []
        self.qt_list = []
        self.qx_list = []
        self.qy_list = []
        self.qz_list = []
        self.lamb_list = []
        self.beta_list = []
        self.nu_list = [] 
        self.omega_list = []
        self.z_list = []
        
        for i in range(n):
            self.Update(dt)
            self.w_list.append(self.w_)
            self.qt_list.append(self.attitude.w)
            self.qx_list.append(self.attitude.x)
            self.qy_list.append(self.attitude.y)
            self.qz_list.append(self.attitude.z)
            self.lamb_list.append(self.lamb)
            self.beta_list.append(self.beta)
            self.nu_list.append(self.nu)
            self.omega_list.append(self.omega)
            self.z_list.append(self.z_)

########################## PLOTS ################################
            
def Plot3DZ(satellite, dt, n):
    satellite.Reset()
    z_list = []
    for i in range(n):
        satellite.Update(dt)
        z_list.append(satellite.z_)
    
    z_listx = [i[0] for i in z_list]
    z_listy = [i[1] for i in z_list]
    z_listz = [i[2] for i in z_list]
    
    
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(z_listx, z_listy, z_listz,'--', label='Z vector rotation')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')
    
    plt.show()      
     
def Plot3DW(satellite, dt, n):
    satellite.Reset()
    w_list = []
    for i in range(n):
        satellite.Update(dt)
        w_list.append(satellite.w_)
    
    w_listx = [i[0] for i in w_list]
    w_listy = [i[1] for i in w_list]
    w_listz = [i[2] for i in w_list]
    
    
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(w_listx, w_listy, w_listz,'--', label='W: inertial rotation vector')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')
    
    plt.show()

def PlotAttitude(satellite, dt, n):
    satellite.Reset()
    t = np.arange(0, dt*n, dt)
    qt_list = []
    qx_list = []
    qy_list = []
    qz_list = []
    for i in range(n):
        satellite.Update(dt)
        qt_list.append(satellite.attitude.w)
        qx_list.append(satellite.attitude.x)
        qy_list.append(satellite.attitude.y)
        qz_list.append(satellite.attitude.z)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.subplots_adjust(left=0.2, wspace=0.6)
    
    ax1.plot(t, qt_list,'ro--')
    ax1.set(title='Attitude components wrt time', ylabel='qw')

    ax2.plot(t, qx_list, 'bo--')
    ax2.set_ylabel('qx')

    ax3.plot(t, qy_list, 'go--')
    ax3.set_ylabel('qy')
    
    ax4.plot(t, qz_list, 'ko--')
    ax4.set_ylabel('qz')
    
    plt.show()
    
    
def PlotLatLong(satellite, dt, n):
    lat_list = [] #beta
    long_list = [] #lambda
    for i in range(n):
        satellite.Update(dt)
        long_list.append(satellite.l)
        lat_list.append(satellite.beta)
    plt.figure()
    plt.plot(long_list, lat_list, 'bo--')
    plt.ylabel('Ecliptic Lattitude (rad)')
    plt.xlabel('Ecliptic Longitude (rad)')
    plt.title('Revolving scanning')
    plt.show()
    

    
    
        
    
        
        

    
    
    
    
        
        
        
        
        
        
