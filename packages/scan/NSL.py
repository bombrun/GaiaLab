#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:59:19 2018

@author: vallevaro
"""

from quaternion import*
from sympy import*
#from frameRotation import*
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
    
        
class Satellite: 

    def __init__(self,*args):  
        self.t = 0
        self.Parameters(*args)
        self.Reset()      
        #self.scanner = Scanner() 
        
    def Parameters(self, S = 4.035, epsilon = np.radians(23.26) , xi =  np.radians(55) , wz = 120):
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

    def InitAttitude(self):
        '''
        Called in __init__ method. 
        Initialises satellite.attitude according to initial values of all parameters. 
        '''
        #q1 = Quaternion(np.sin(self.epsilon/2), 0, 0, np.cos(self.epsilon/2))                            #epsilon about l-axis
        #q2 = Quaternion(0,0,np.sin(self.l0/2),np.cos(self.l0/2))                                         #lambda about n-axis
        #q3 = Quaternion(np.sin((self.nu0-np.pi/2.)/2), 0, 0, np.cos((self.nu0-np.pi/2.)/2))              #nu - 90 abbout l-axis
        #q4 = Quaternion(0,np.sin((np.pi/2. - self.xi)/2), 0, np.cos((np.pi/2. - self.xi)/2))             #90-xi about m-axis
        #q5 = Quaternion(0,0, np.sin(self.omega0/2.), np.cos(self.omega0/2.))                             #omega abbout n-axis (z-axis).
        
        #I think the above were given in SAG-LL-30 in the [x,y,z,w] convention, so trying here in the [w,x,y,z]
        
        q1 = Quaternion(np.cos(self.epsilon/2), np.sin(self.epsilon/2), 0, 0)                            #epsilon about l-axis
        q2 = Quaternion(np.cos(self.l0/2), 0,0,np.sin(self.l0/2))                                        #lambda about n-axis
        q3 = Quaternion(np.cos((self.nu0-np.pi/2.)/2), np.sin((self.nu0-np.pi/2.)/2), 0, 0)              #nu - 90 abbout l-axis
        q4 = Quaternion(np.cos((np.pi/2. - self.xi)/2), 0,np.sin((np.pi/2. - self.xi)/2), 0)             #90-xi about m-axis
        q5 = Quaternion(np.cos(self.omega0/2.), 0,0, np.sin(self.omega0/2.))                             #omega abbout n-axis (z-axis).
        
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
        l_,j_,k_ = ljk(self.epsilon)
        self.s_ = l_*np.cos(self.l) + j_*np.sin(self.l)

    def Update(self, dt):
        self.t = self.t + dt
        dL = self.ldot * dt
        self.l = self.l + dL
        
        #Updates Nu
        nudot = self.ldot*(np.sqrt(self.S**2 - np.cos(self.nu)**2) + np.cos(self.xi)*np.sin(self.nu))/np.sin(self.xi)
        dNu = nudot*dt
        self.nu = self.nu + dNu
        
        #LatitudeAngles
        self.lamb_z = self.l + np.arctan(np.tan(self.xi)*np.cos(self.nu))
        self.beta_z = np.arcsin(np.sin(self.xi)*np.sin(self.nu))
        
        #Updates Omega
        omegadot = self.wz - nudot*np.cos(self.xi) - self.ldot*np.sin(self.xi)*np.sin(self.nu)
        dOmega = omegadot * dt
        self.omega = self.omega + dOmega
        
        #calculates coordinates and then calculates s-vector 
        l_,j_,k_ = ljk(self.epsilon)
        #old_s_ = self.s_
        self.s_ = l_*np.cos(self.l) + j_*np.sin(self.l)
        
        #Calculates z-axis from cross product: deltaz = (k x z)*ldot + (s x z)dNu
        zdot_ = np.cross(k_, self.z_)*self.ldot + np.cross(self.s_,self.z_)*nudot
        dz_ = zdot_*dt
        self.z_ = self.z_ + dz_
        self.z_ = self.z_ / np.linalg.linalg.norm(self.z_)        #Renormalise unit vector to prevent expansion over time.
        
        #updates inertial rotation vector
        self.w_ = k_*self.ldot + self.s_*nudot + self.z_*omegadot
        
        #calculates new attitude by deltaquat
        w_magnitude = np.sqrt((self.w_[0])**2 + (self.w_[0])**2 + (self.w_[0])**2)
        dzheta = w_magnitude * dt
        deltaquat = rotation_quaternion(self.w_, dzheta)
        self.attitude = deltaquat*self.attitude
        
    def GetAttitude(self, time, dt = 0.001):
        self.Reset()
        for i in np.arange(time/dt):
            self.Update(dt)
        self.Update(time%dt)
        return self.attitude
        
    def GetXAxis(self, time):
        attitude = self.GetAttitude(time)
        x_quat = attitude * Quaternion(0,1,0,0) * attitude.conjugate() 
        x_ = quaternion_to_vector(x_quat)
        return x_
    
    
        
################################ SCANNER ######################################
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
        
class Scanner:
    '''
    __init__
    
    tolerance: [float][radians] for searching a star position with scanner axis.
    
    .Reset()
    .GetXAxis(attitude): [quaternion]
        returns: [3D vector]
    '''
    def __init__(self, tolerance):
        self.observations = []
        self.axis  = []
        self.tolerance = tolerance
        
    def Reset(self):
        self.observations = []
        self.axis  = []
        self.times_obs = []
 
    def GetXAxis(self, attitude):
        x_quat = attitude * Quaternion(0,1,0,0) * attitude.conjugate() 
        x_ = quaternion_to_vector(x_quat) 
        return x_  
        
    def Inst_Scan(self, sky, attitude):
        x_ = self.GetXAxis(attitude)
        found_stars = []
        for star in sky.elements:
            x_ = unit_vector(x_)
            angle = np.arccos(np.dot(star.coor, x_)) #are both normalised?
            if np.abs(angle) < self.tolerance:
                found_stars.append([star, angle])
                self.axis.append(x_)
                
        return found_stars, x_
        
                                                                                
def Do_Scan(scanner, satellite, sky, time_total, dt):
    scanner.Reset()
    satellite.Reset()
    n_steps = time_total/dt
    for i in np.arange(n_steps):
        satellite.Update(dt)
        found_stars, x_ = scanner.Inst_Scan(sky, satellite.attitude)
        for info in found_stars:
            scanner.observations.append(info)
            scanner.times_obs.append(i*dt)


def Angle(scanner):
    angles = [i[1] for i in scanner.observations]
    plt.figure()
    plt.grid()
    plt.plot(angles, 'bo')
    plt.ylabel('rad')
    plt.title('Angle between Scanner Axis and star position')
    plt.show()
    
    
def PlotAxis(scanner):
    
    x = [i[0] for i in scanner.axis]
    y = [i[1] for i in scanner.axis]
    z = [i[2] for i in scanner.axis]
    
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x, y, z,'bo', label='x-axis')

    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    
    plt.show()      
    
def PlotObs(scanner,satellite):
    stars_list = []
    for t in scanner.times_obs:
        star_ = satellite.GetXAxis(t)
        stars_list.append(star_)
        
    x = [i[0] for i in stars_list]
    y = [i[1] for i in stars_list]
    z = [i[2] for i in stars_list]
        

    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x, y, z,'b*', label='Measurement')

    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    
    plt.show()         
    
 ############################################################################    
                          
def Plot3DX(satellite, dt, n):
    satellite.Reset()
    for i in np.arange(n/dt):
        satellite.Update(dt)
        x_quat = satellite.attitude * Quaternion(0,1,0,0) * satellite.attitude.conjugate() 
        x_ = quaternion_to_vector(x_quat)
        x_list.append(x_)
    
    x_listx = [i[0] for i in x_list]
    x_listy = [i[1] for i in x_list]
    x_listz = [i[2] for i in x_list]
    
    
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x_listx, x_listy, x_listz,'--', label='X vector rotation')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    plt.show()               


    
