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
    """
    Creates n source objects from random numbers.
    Args:
        n (int): number of sources to be created.
    Attributes:
        elements (list of obj): list of source objects.
    """

    def __init__(self, n):
        self.elements = []
        for i in range(n):      # n number of objects in the time-interval given
            alpha = np.random.uniform(0,2*np.pi)       # time of possible observation in a day duration.
            delta = np.random.uniform(-np.pi/2, np.pi/2)
            mualpha = np.random.uniform(0,0.1)
            mudelta = np.random.uniform(0,0.1)
            self.elements.append(Source(alpha, delta, mualpha, mudelta))
            

class Source:
    """
    Defines source star by horizontal coordinate system parameters.
    Args:
        alpha (float): galactic longitudinal angle [rad].
        delta (float): galactic altitude angle (rad),
        mualpha (float): velocity in alpha of source [mas/yr].
        mudelta (float): velocity in delta of source [mas/yr].
        parallax (float): stellar parallax angle from annual parallax [mas].

    Attributes:
        coor (np.dnarray): (alpha, delta) horizontal coordinate system.
        velocity (np.dnarray): (mualpha, mudelta, parallax) velocity and position parameters.
    """

    def __init__(self, alpha, delta, mualpha, mudelta, parallax = 1):
        self.coor = xyz(alpha, delta)
        self.velocity = np.array([mualpha, mudelta, parallax])


class Satellite: 
    """
    Satellite that moves changing its attitude according to Nominal Scanning Law.
    Attributes:
        t (float): initial time of satellite [days].
        storing_list (list of objects): first empty, filled when satellite.move() method is called.
    """
    def __init__(self, *args):
        self.init_parameters(*args)
        self.init_attitude()
        self.reset()
        self.storing_list = []
        self.move(0, 365*5, 0.1)
        
    def init_parameters(self, S=4.036, epsilon=np.radians(23.26), xi=np.radians(45), wz=120,
                        ldot=2*np.pi/365, t0=0, nu0=0, omega0=0, l0=0, beta0=0):
        """
        Sets satellite to initialization status.
        Args:
            S (float): -> dz/dlambda; change in z-axis of satellite with respect to solar longitudinal angle.
            epsilon (float): ecliptical angle [rad].
            xi (float): revolving angle [rad].
            wz (float): z component of inertial spin vector [arcsec/s].
            ldot (float): velocity of the sun around the earth [rad/day].
            t0 (float): J2000 [days].
            nu0 (float): initial revolving angle at t0 [rad].
            omega0 (flaot): initial spin angle at t0 [rad].
            l0 (float): initial solar longitud angle at t0 [rad].
            beta0 (float): initial solar altitude angle at t0 [rad].
        """

        # Change of units
        wz = wz/206264.8062470946 #to radians/s
        wz = wz*60*60*24. #to radians/days
        # Constant parameters
        self.S = S
        self.epsilon = epsilon
        self.xi = xi
        self.wz = wz
        
        # Initial-value conditions for functions.
        self.ldot = ldot
        self.t = t0
        
        self.nu = nu0
        self.omega = omega0
        self.lamb = l0
        self.beta = beta0

    def init_attitude(self):
        '''
        Initialises satellite.attitude according to initial values of all parameters.
        Notes:
            always needs to have parameters init_parameters called first, otherwise it would calculate the attitude
            from current parameters values.
        '''
        q1 = Quaternion(np.cos(self.epsilon/2), np.sin(self.epsilon/2), 0, 0)                   # epsilon about l-axis
        q2 = Quaternion(np.cos(self.lamb/2), 0,0,np.sin(self.lamb/2))                             # lambda about n-axis
        q3 = Quaternion(np.cos((self.nu-np.pi/2.)/2), np.sin((self.nu-np.pi/2.)/2), 0, 0)     # nu - 90 about l-axis
        q4 = Quaternion(np.cos((np.pi/2. - self.xi)/2), 0,np.sin((np.pi/2. - self.xi)/2), 0)    # 90-xi about m-axis
        q5 = Quaternion(np.cos(self.omega/2.), 0,0, np.sin(self.omega/2.))                    # omega about z-axis.
        
        q_total = q1*q2*q3*q4*q5
        self.attitude = q_total

    def reset(self):
        '''
        Resets satellite to initialization status.
        '''
        # Initialization of satellite.attitude.
        self.init_parameters()
        # Initial orientation values set equal to initial-value conditions.
        self.init_attitude()

        # Initialization of z-axis and inertial rotation vector.
        z_quat = self.attitude * Quaternion(0,0,0,1) * self.attitude.conjugate()
        self.z_ = np.array([z_quat.x, z_quat.y, z_quat.z])
        self.w_ = np.cross(np.array([0,0,1]),self.z_)
        l_,j_,k_ = ljk(self.epsilon)
        self.s_ = l_*np.cos(self.lamb) + j_*np.sin(self.lamb)

    def update(self, dt):
        """
        Calculates the change in all parameters of satellite to update its attitude.
        First order approximation (Euler's method) to calculate derivatives of the variables.
        Args:
            dt (float): delta time to update satellite's parameters from initial time (self.t) to self.t + dt
        Attributes:
            t (float): time
            lamb (float): solar longitudinal angle in galatic frame.
            nu (float): revolving phase.
            lamb_z (float): solar longitudinal angle in ecliptic plane.
            beta_z (float): solat altitude angle in ecliptic plane.
            l_, j_, k_ (np.dnarray): ecliptic triad frame.
            s_ (np.dnarray): vector to position of the sun from satellite's centre.
            z_ (np.dnarray): z-axis of SRS frame wrt BCRS.
            x_ (np.dnarray): x-axis of SRS frame wrt BCRS.
            w_ (np.dnarray): inertial spin vector wrt BCRS.
            attitude (quaternion): attitude updated after dt time change, from initialization parameters.

        """
        self.t = self.t + dt
        dL = self.ldot * dt
        self.lamb = self.lamb + dL
        
        # Updates Nu
        nudot = self.ldot*(np.sqrt(self.S**2 - np.cos(self.nu)**2) + np.cos(self.xi)*np.sin(self.nu))/np.sin(self.xi)
        dNu = nudot*dt
        self.nu = self.nu + dNu
        
        # LatitudeAngles
        self.lamb_z = self.lamb + np.arctan(np.tan(self.xi)*np.cos(self.nu))
        self.beta_z = np.arcsin(np.sin(self.xi)*np.sin(self.nu))
        
        # Updates Omega
        omegadot = self.wz - nudot*np.cos(self.xi) - self.ldot*np.sin(self.xi)*np.sin(self.nu)
        dOmega = omegadot * dt
        self.omega = self.omega + dOmega
        
        # Calculates coordinates and then calculates s-vector
        l_,j_,k_ = ljk(self.epsilon)
        self.s_ = l_*np.cos(self.lamb) + j_*np.sin(self.lamb)
        
        # Calculates z-axis from cross product: deltaz = (k x z)*ldot + (s x z)dNu
        zdot_ = np.cross(k_, self.z_)*self.ldot + np.cross(self.s_,self.z_)*nudot
        dz_ = zdot_*dt
        self.z_ = self.z_ + dz_
        self.z_ = self.z_ / np.linalg.linalg.norm(self.z_)
        
        # Updates inertial rotation vector
        self.w_= k_*self.ldot + self.s_*nudot + self.z_*omegadot
        
        # Calculates new attitude by delta_quat
        w_magnitude = np.sqrt((self.w_[0])**2 + (self.w_[0])**2 + (self.w_[0])**2)
        d_zheta = w_magnitude * dt
        deltaquat = rotation_quaternion(self.w_, d_zheta)
        self.attitude = deltaquat*self.attitude
        
        # x axis
        x_quat = self.attitude * Quaternion(0,1,0,0) * self.attitude.conjugate()
        self.x_= quaternion_to_vector(x_quat)

    def move(self, ti, tf, dt):
        '''
        Creates data necessary for step numerical methods performed in builtin method .update()
        Args:
            ti (float): integrating time lower limit [days]
            tf (float): integrating time upper limit [days]
            dt (float): step discretness of integration.
        Notes:
            The data is stored in satellite.storing_list
        '''

        self.dt = dt
        self.t = ti
        n_steps = (tf-ti)/dt
        for i in np.arange(n_steps):
            self.update(dt)
            self.storing_list.append([self.t, self.lamb, self.nu,
                                    self.omega, self.w_,self.s_, 
                                    self.z_, self.attitude, self.x_])
    
        self.storing_list.sort(key=lambda x: x[0])

    def reset_to_time(self, t):
        '''
        Resets satellite to time t, along with all the parameters corresponding to that time.
        Args:
            t (float): time from J2000 [days]
        '''
        if t == 0:
            self.reset()
        else:
            templist = [obj for obj in self.storing_list if obj[0] <= t]
            list_element = templist[-1]
            self.t = list_element[0]
            self.lamb = list_element[1]
            self.nu = list_element[2]
            self.omega = list_element[3]
            self.w_ = list_element[4]
            self.s_ = list_element[5]
            self.z_ = list_element[6]
            self.attitude = list_element[7]
            self.x_ = list_element[8]

            # Account for decimal loss difference from discreteness from NM.
            deltat = t%self.t
            self.update(deltat)
      
    def get_attitude(self, t):
        """
        :param t: time at which want to get attitude of satellite.
        :return: quaternion
        """
        self.reset_to_time(t)
        return self.attitude
        
    def get_xaxis(self, t):
        """
        :param t: time at which want to get xaxis of satellite wrt BCRS.
        :return: np.dnarray
        """
        self.reset_to_time(t)
        return self.x_
       

class Scanner:
    """
    Args:
        ccd (float): width of the telescope field of view.
        delta_z (float): height of the telescope field of view.
        delta_y (float): width of the line of intercept.

    Attributes:
        times_to_scan_star (list of floats): times where star within CCD field of view.
        obs_times (list of floats): times from J2000 at which the star is inside of the line of intercept.
        stars_positions (list of arrays): positions calculated from obs_times of transits using satellite's attitude.
    """
    def __init__(self, ccd=0.2, delta_z=0.15, delta_y=0.01):
        self.delta_z = delta_z
        self.delta_y = delta_y
        self.ccd = ccd
        
        self.obs_times = []
        self.stars_positions = []
        self.times_to_scan_star = []
        
    def reset(self):
        """
        :return: empty all attribute lists from scanner before beginning new scanning period.
        """
        self.obs_times = []
        self.stars_positions = []
        self.times_to_scan_star = []
   
    def intercept(self, star, satellite):
        """
        :param star: source object.
        :param satellite: satellite object.
        :return: populates scanner.times_to_scan list, before deep_scan is executed.
        """
        for obj in satellite.storing_list:
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
                    self.times_to_scan_star.sort()

    def deep_scan(self, satellite, deep_dt=0.001):
        """
        Increases precision of satellite at points where source is intercept by scanner in the CCD.
        :param satellite: satellite object.
        :param deep_dt: new step dt fur higher numerical method precision.
        """
        for t in self.times_to_scan_star:
            satellite.move(t-0.5, t+0.5, deep_dt)

def star_finder(satellite, scanner, sky):
    """
    Finds times at which source transit CCD line of scanner and estimates their position in the BCRS frame.
    :param satellite: Satellite object.
    :param scanner: Scanner object.
    :param sky: Sky object.
    """
    scanner.reset()
    for star in sky.elements:
        scanner.intercept(star, satellite)
        scanner.deep_scan(satellite)
        
        for obj in satellite.storing_list:
            x_telescope1 = obj[8]  #wrt bcrs frameß
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


def run():
    start_time = time.time()
    sky = Sky(1)
    scan = Scanner()
    gaia = Satellite()
    gaia.storing_list = []
    gaia.move(0,365*3, 0.1)
    
    star_finder(gaia, scan, sky)

    seconds = time.time() - start_time
    m, s = divmod(seconds, 365)
    h, m = divmod(m, 60)

    print ("%d:%02d:%02d" % (h, m, s))
    print (len(scan.stars_positions))
    return scan, gaia, sky

    


