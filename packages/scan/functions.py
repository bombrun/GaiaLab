from quaternion import*
from sympy import*
#from frameRotation import*
from sympy import Line3D, Point3D
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import datetime

################################## FUNCTIONS ##################################
                                                                                                                                       
def vector(x,y,z): 
    return np.array([x,y,z])

def unit_vector(vector): 
    return vector / np.linalg.norm(vector) 
            
def vector_to_point(vector):
    return Point3D(vector[0], vector[1], vector[2])
    
def point_to_vector(point):          
    return np.array([point.x, point.y, point.z])           

def vector_to_quaternion(vector):
    return Quaternion(0, float(vector[0]), float(vector[1]), float(vector[2]))  
    
def quaternion_to_vector(quat):
    return np.array([quat.x,quat.y,quat.z])
       
def rotation_quaternion(vector, angle): 
    '''    
    Calculates Quaternion equivalent to a rotation given by a vector and a angle in radians.
    '''   
    norm = np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
    vector = vector/norm
    t = np.cos(angle/2.)
    x = np.sin(angle/2.)*vector[0]
    y = np.sin(angle/2.)*vector[1]
    z = np.sin(angle/2.)*vector[2]
    
    qvector = Quaternion(t,x,y,z)
    return qvector   
    
def alphadelta(vector):
    alpha = np.arctan2(vector[1], vector[0])
    delta = np.arctan2(vector[2], np.sqrt(vector[1]**2 + vector[0]**2))  
    return alpha, delta
    
def xyz(azimuth, altitude):
    '''
    args
    _____
    
    azimuth: [float]
    altitude: [float]
    
    Returns
    ________
    
    vector = (x,y,z)
    
    '''
    x = np.cos(azimuth)*np.cos(altitude)
    y = np.sin(azimuth)*np.cos(altitude)
    z = np.sin(altitude)
    return np.array([x,y,z]) 
    
def ljk(epsilon):
    
    l = np.array([1,0,0])
    j = np.array([0, np.cos(epsilon), np.sin(epsilon)])
    k = np.array([0, -np.sin(epsilon), np.cos(epsilon)])
    return l,j,k

def matrix_attitude(q):
    q = q.unit()
    q1, q2, q3, q4 = q.w, q.x, q.y, q.z
    a11 = q1**2 - q2**2 - q3**2 + q4**2
    a12 = 2*(q1*q2 - q3*q4)
    a13 = 2*(q1*q3 - q2*q4)
    a21 = 2*(q1*q2 - q3*q4)
    a22 = -q1**2 + q2**2 - q3**2 + q4**2
    a23 = 2*(q2*q3 + q1*q4)
    a31 = 2*(q1*q3 + q2*q4)
    a32 = 2*(q2*q3 - q1*q4)
    a33 = -q1**2 -q2**2 +q3**2 + q4**2
    
    A = np.array([a11,a12,a13,a21,a22,a23,a31,a32,a33])
    
    A = A.reshape((3,3))
    return A
     
def W_check(self, t):
        '''
        Args
        ______
        t: [float] time
        
        Returns
        ________
        Omega: [array] (3,1) spin phase. Time dependet, but with constant inertial rotation rate about zaxis in SRS. 
        
        '''
        wx = 2 * np.array([q4, q3, -q2, -q1])*diff(self.attitude)
        wy = 2 * np.array([-q3, q4, q1, -q2])*diff(self.attitude)
        wz = 2* np.array([q2, -q1, q4, -q3])*diff(self.attitude)
        
def Lambda(self, t = None):
    '''
    Args
    ______
    t: [float] time
    
    Returns
    ________
    Lambda: [float] nominal longitud of the sun in elliptical plane. Time dependent, simple analytical function.
    ''' 
    
    if t == None:
        t = Symbol('t')
        lanbda = t* (2*np.pi/365)
        return lanbda 
    if t != None:
        lanbda = t* (2*np.pi/365)
        return lanbda 
    
def Nu(self, t0 = 1., tf = 10., n=101):
    '''
    Args
    ______
    t: [float] time
    
    Returns
    _______
    nu: [float] revolving phase. Time dependent, give a nearly constant precession rate.
    '''
    #integrate expression to get nu.
    nu = Symbol('nu')
    t = Symbol('t')

    deltat = (tf-t0)/(n-1)
    t  = np.linspace(t0, tf, n)
    nu = np.zeros([n])
    nu[0] = self.nu0
    for i in range(1,n):
        nu[i] = deltat*(self.ldot + (np.sqrt(self.S**2- (np.cos(nu[i-1]))**2) + np.cos(self.xi)*np.sin(nu[i-1]))/np.sin(self.xi)) + nu[i-1]
    
    return t, nu  
    
def Omega(self, t0 = 1., tf = 10., n=101):
        t, nu = self.Nu(t0 = 1., tf = 10., n=101)
                
        w = Symbol('w')
        t = Symbol('t')
        
        t0 = 1.
        tf = 10.
        w0 = 0.5
        n = 101
        deltat = (tf-t0)/(n-1)
        
        t  = np.linspace(t0, tf, n)
        w = np.zeros([n])
        w[0] = w0
        for i in range(1,n):
            w[i] = self.wz - dnu[i]*np.cos(self.xi) - self.ldot*np.sin(self.xi)*np.sin(nu[i-1])
        return t, w 
        
def WMatrix(self, w_):
        w_[0] = wl
        w_[1] = wm
        w_[2] = wn
        
        
        wMatrix = np.array([0, -wl,-wm,-wn,wl,0,wn,-wm,wm,-wn,0,wl,wn,wm,-wl,0])   #need to change this to have proper structure of quaternion (x,y,z,t) rather than (t,x,y,z)
        wMatrix = wMatrix.reshape(4,4)
       
        return wMatrix  
          
def SRS(attitude, vector):
    '''
    Changes coordinates of a vector in BCRS to SRS frame.
    '''
    q_vector_bcrs= vector_to_quaternion(vector)
    q_vector_srs = attitude * q_vector_bcrs * attitude.conjugate()  
    return np.array([q_vector_srs.x, q_vector_srs.y, q_vector_srs.z])    
   
def BCRS(attitude, vector):
    '''
    Changes coordinates of a vector in SRS to BCRS frame.
    '''
    q_vector_srs= vector_to_quaternion(vector)
    q_vector_bcrs = attitude.conjugate() * q_vector_srs * attitude  
    return np.array([q_vector_bcrs.x, q_vector_bcrs.y, q_vector_bcrs.z])