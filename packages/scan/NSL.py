#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:59:19 2018

@author: vallevaro
"""

from quaternion import*
from sympy import*
from functions import*
from plots import*

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
    
################################ SKY ######################################
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
        self.Parameters(*args)
        self.Reset()
        self.storinglist = []     
        
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
        
        #initial-value conditions for functions.
        self.ldot = (2*np.pi/365)
        self.t0 = 0.
        
        self.nu0 = 0.
        self.omega0 = 0.
        self.l0 = 0.
        self.beta0 = 0.

    def InitAttitude(self):
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
        
    def Reset(self):
        '''
        Resets satellite to t = 0 and initialization status.
        '''
        
        #initialization of satellite.attitude.
        #self.storinglist = []
        self.attitude = self.InitAttitude()
        
        #Initial orientation values set equal to initial-value conditions.
        self.t = self.t0
        self.nu = self.nu0
        self.omega = self.omega0
        self.l = self.l0
        self.beta = self.beta0
        
        #Initialization of z-axis and inertial rotation vector.
        z_quat = self.attitude * Quaternion(0,0,0,1) * self.attitude.conjugate()
        self.z_ = np.array([z_quat.x, z_quat.y, z_quat.z])
        self.w_ = np.cross(np.array([0,0,1]),self.z_)
        l_,j_,k_ = ljk(self.epsilon)
        self.s_ = l_*np.cos(self.l) + j_*np.sin(self.l)

    def Update(self, dt):
        self.t = self.t + dt
        self.dL = self.ldot * dt
        self.l = self.l + self.dL
        
        #Updates Nu
        self.nudot = self.ldot*(np.sqrt(self.S**2 - np.cos(self.nu)**2) + np.cos(self.xi)*np.sin(self.nu))/np.sin(self.xi)
        self.dNu = self.nudot*dt
        self.nu = self.nu + self.dNu
        
        #LatitudeAngles
        self.lamb_z = self.l + np.arctan(np.tan(self.xi)*np.cos(self.nu))
        self.beta_z = np.arcsin(np.sin(self.xi)*np.sin(self.nu))
        
        #Updates Omega
        omegadot = self.wz - self.nudot*np.cos(self.xi) - self.ldot*np.sin(self.xi)*np.sin(self.nu)
        dOmega = omegadot * dt
        self.omega = self.omega + dOmega
        
        #calculates coordinates and then calculates s-vector 
        l_,j_,k_ = ljk(self.epsilon)
        self.s_ = l_*np.cos(self.l) + j_*np.sin(self.l)
        
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

    
    def Storing(self,t0, tf, dt):
        
        if len(self.storinglist) != 0:
            raise Exception('Careful, already made storing list')
                
        self.t0 = t0
        self.tf = tf
        n_steps = (tf-t0)/dt
        self.t = t0
        for i in np.arange(n_steps):
            self.Update(dt)
            self.storinglist.append([self.t, self.l, self.nu,self.omega, self.w_,self.s_, self.z_, self.attitude, self.x_])
        self.storinglist.sort()
        
    def ResetToTime(self, t):

        templist = [obj for obj in self.storinglist if obj[0] <= t]
        if len(templist) != 0:
            listelement = templist[-1]
        
            self.t        = listelement[0]
            self.l        = listelement[1]
            self.nu       = listelement[2]
            self.omega    = listelement[3]
            self.w_       = listelement[4]
            self.s_       = listelement[5]
            self.z_       = listelement[6]
            self.attitude = listelement[7]
            self.x_       = listelement[8]
            
            
        else: 
            self.Reset()
                
        #LatitudeAngles
        self.lamb_z = self.l + np.arctan(np.tan(self.xi)*np.cos(self.nu))
        self.beta_z = np.arcsin(np.sin(self.xi)*np.sin(self.nu))  
        
    def GetAttitude(self, t): 
        self.ResetToTime(t)
        return self.attitude
        
    def GeXAxis(self, t):
        self.ResetToTime(t)
        x_quat = self.attitude * Quaternion(0,1,0,0) * self.attitude.conjugate() 
        x_ = quaternion_to_vector(x_quat)
        return x_
       
################################ SCANNER ######################################        
class Scanner:
    '''
    __init__
    
    tolerance: [float][radians] for searching a star position with scanner axis.
    
    .Reset()
    .xxis(attitude): [quaternion]
        returns: [3D vector]
    '''
    def __init__(self, tolerance, dAngle = np.radians(106.5)):
        self.dAngle = dAngle
        self.tolerance = tolerance
        self.storinglist = []
        
    def Reset(self):
        self.storinglist = []
    
    
    def SelectRange(self, t0_scan, tf_scan, satellite):
        self.storinglist = [x for x in satellite.storinglist if x[0]>= t0_scan and x[0] <= tf_scan]
    
    def SortPositions(self):
        self.starspositions = []
        for obj in self.storinglist:
            x_ = obj[8]
            self.starspositions.append(x_)
          
###########################################################################  
 
def StarFinder(satellite, scanner, t0, tf, sky, dt):
    satellite.Reset()
    scanner.Reset()
    satellite.storinglist =[]
    satellite.Storing(t0, tf, dt)
    diff_list = []
    for star in sky.elements:
        for idx, obj in enumerate(satellite.storinglist):
            diff_ = np.abs(obj[8] - star.coor)
            diff_list.append(mag(diff_))
            if mag(diff_) < scanner.tolerance:
                scanner.storinglist.append(obj)

        
def DeepScan(star, scanner, satellite, tinitial, tfinal, dt = 0.001):
    
    satellite.ResetToTime(tinitial)
    
    n_steps = (tfinal - tinitial)/dt
    #t = obj[0]
    objnext = scanner.storinglist[idx+1]
    
    attitude_objnext = objnext[7]
    attitude_obj= obj[7]
    #x_1 = scanner.GetFV1(attitude)
    #x_2 = scanner.GetFV2(attitude)
    z_1 = obj[6]
    for star in sky.elements:
        if np.arccos(np.dot(star.coor, z_1)) <= (np.pi/2 - scanner.tolerance) and np.arccos(np.dot(star.coor, z_1)) <= (np.pi/2 + scanner.tolerance): 
            star_coorquat_telescope1 = attitude_obj * vector_to_quaternion(star.coor) * attitude_obj.conjugate()
            star_coor_telescope1 = quaternion_to_vector(star_coorquat_telescope1)
            star_coorquat_telescope1_next = attitude_objnext * vector_to_quaternion(star.coor) * attitude_objnext.conjugate()
            star_coor_telescope1_next = quaternion_to_vector(star_coorquat_telescope1_next)
            
            
            q2 = rotation_quaternion(np.array([0,0,1]), scanner.dAngle)
            star_coor_telescope2 = quaternion_to_vector(q2 * star_coorquat_telescope1 * q2.conjugate())
            star_coor_telescope2_next = quaternion_to_vector(q2 * star_coorquat_telescope1_next * q2.conjugate())
            
            if star_coor_telescope1[1] < 0 and star_coor_telescope1_next[1] > 0:
                found_stars_1.append(obj)
            if star_coor_telescope2[1] < 0 and star_coor_telescope2_next[1] > 0:
                found_stars_2.append(obj)      
    return found_stars_1, found_stars_2
    
    

###########################################################################
        
def Run():
    start_time = time.time()
    
    scan = Scanner(0.01)
    gaia = Satellite()
    sky = Sky(1)
    true_stars = Do_Scan(scan, gaia, sky, 365, 0.001)
    PlotObsTwoTelescopes(scan, true_stars)
    
    seconds = time.time() - start_time
    m, s = divmod(seconds, 365)
    h, m = divmod(m, 60)
    
    print ("%d:%02d:%02d" % (h, m, s))
    return scan, gaia, sky, true_stars

    
###########################################################################
def PlotObsTwoTelescopes(scanner, true_stars):
    stars_position_1 = scanner.observations_1
    x_1= [i[0] for i in stars_position_1]
    y_1= [i[1] for i in stars_position_1]
    z_1= [i[2] for i in stars_position_1]
    
    stars_position_2 = scanner.observations_2
    x_2= [i[0] for i in stars_position_2]
    y_2= [i[1] for i in stars_position_2]
    z_2= [i[2] for i in stars_position_2]
    
    sx= [i.coor[0] for i in true_stars]
    sy= [i.coor[1] for i in true_stars]
    sz= [i.coor[2] for i in true_stars]
    

    fig, ((ax1, ax2, ax3)) = plt.subplots(1,3)
    fig.subplots_adjust(left=0.2, wspace=0.6)
    
    ax1.plot(x_1, y_1,'ro', label='FV1')
    ax1.plot(x_2, y_2, 'go', label='FV2')
    ax1.plot(sx, sy, 'b*', label='True Star Position')
    ax1.legend()
    ax1.set(xlabel='XY PLANE')
    
    ax2.plot(y_1, z_1, 'ro')
    ax2.plot(y_2, z_2, 'go')
    ax2.plot(sy, sz, 'b*')
    ax2.set(xlabel='YZ PLANE')

    ax3.plot(x_1, z_1, 'ro')
    ax3.plot(x_2, z_2, 'go')
    ax3.plot(sx, sz, 'b*')
    ax3.set(xlabel='XZ PLANE')
    
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x_1, y_1, z_1,'ro', label='FV1')
    ax.plot(x_2, y_2, z_2,'go', label='FV2')
    ax.plot(sx, sy, sz,'b*', label='True Star Position')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')
    
    plt.show()    

    plt.show()
    
