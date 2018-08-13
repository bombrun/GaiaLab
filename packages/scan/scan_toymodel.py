# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:59:19 2018

@author: vallevaro

http://docs.astropy.org/en/stable/coordinates/index.html#module-astropy.coordinates
"""

import frame_transformations as ft
from quaternion import Quaternion
import numpy as np
import time
from scipy import interpolate
from scipy import optimize

from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


#vega = Source("vega", 279.2333, 38.78, 128.91, 201.03, 286.23, -13.9)
#proxima = Source("proxima",217.42, -62, 768.7, 3775.40, 769.33, 21.7)
#sirio = Source("sirio", 101.28, -16.7161, 379.21, -546.05, -1223.14, -7.6)
#alphacentauri = Source('alpha centauri', )
#barnard = Source("Barnard", )
#wolf = Source('Wolf 359', )
#lalande = Source("Lalande 21185", )
#luyten = Source('Luyten 726-8B', )
#ross154 = Source(
#demo = Source("demo", 217.42, -62, 1, -1700, -2000, -30)

class Source: #need to take care of units

    def __init__(self, name, alpha0, delta0, parallax, mu_alpha, mu_delta, mu_radial):
        """
        :param alpha0: deg
        :param delta0: deg
        :param parallax: mas
        :param mu_alpha: mas/yr
        :param mu_delta: mas/yr
        :param mu_radial: km/s
        """
        self.name = name
        self.init_param(alpha0, delta0, parallax, mu_alpha, mu_delta, mu_radial)
        self.alpha = self.__alpha0 #rad
        self.delta = self.__delta0 #rad

    def init_param(self, alpha0, delta0, parallax, mu_alpha, mu_delta, mu_radial):
        self.__alpha0= np.radians(alpha0) #rad
        self.__delta0=  np.radians(delta0) #rad
        self.parallax = parallax    #mas
        self.mu_alpha_dx = mu_alpha*np.cos(self.__delta0)   #mas/yr
        self.mu_delta = mu_delta                       #mas/yr
        self.mu_radial = mu_radial    #mas*km/s

        self.direction_bcrs= ft.to_direction(self.__alpha0, self.__delta0)
        self.coor_bcrs = ft.to_cartesian(self.__alpha0, self.__delta0, self.parallax)

    def reset(self):
        self.alpha = self.__alpha0
        self.delta = self.__delta0

    def set_time(self, t): #implement here relativistic effects
        mu_alpha_dx = self.mu_alpha_dx * 4.8473097e-9 / 365 #from mas/yr to rad/day
        mu_delta = self.mu_delta * 4.848136811095e-9 / 365  #from mas/yr to rad/day
        self.alpha = self.__alpha0 + mu_alpha_dx*t  #rad
        self.delta = self.__delta0 + mu_delta*t     #rad

    def barycentric_vector(self, t):
        """
        alpha: rad
        delta: rad
        parallax: mas
        :param t: days
        :return: vector in parsecs
        """
        self.set_time(t)
        u_bcrs = ft.to_cartesian(self.alpha, self.delta, self.parallax)
        return u_bcrs

    def topocentric_function(self, satellite):
        """
        :param satellite:
        :return: lambda function of the position of the star from the satellite´s lmn frame.
        """
        p, q, r = ft.pqr(self.alpha, self.delta)
        mastorad = 2*np.pi/(1000*360*3600)
        kmtopc = 3.24078e-14
        sectoday = 3600*24
        AUtopc = 4.8481705933824e-6

        mu_alpha_dx = self.mu_alpha_dx*mastorad/365   #mas/yr to rad/day
        mu_delta = self.mu_delta*mastorad/365  #mas/yr to rad/day
        mu_radial = self.parallax*mastorad*self.mu_radial*kmtopc*sectoday #km/s to aproximation rad/day

        topocentric_function = lambda t: self.barycentric_vector(0) + t*(p*mu_alpha_dx+ q*mu_delta + r*mu_radial) - satellite.ephemeris_bcrs(t)*AUtopc
        return topocentric_function

    def topocentric_angles(self, satellite, t):
        """

        :param satellite:
        :param t: [days]
        :return: delta alpha, delta delta [mas][mas]
        """
        mastorad = 2 * np.pi / (1000 * 360 * 3600)
        u_lmn_unit = self.topocentric_function(satellite)(t)/np.linalg.norm(self.topocentric_function(satellite)(t))
        alpha_obs, delta_obs, radius = ft.to_polar(u_lmn_unit)
        if alpha_obs < 0:
            alpha_obs = alpha_obs + 2*np.pi
        delta_alpha_dx_mas = (alpha_obs - self.__alpha0) * np.cos(self.__delta0) / mastorad
        delta_delta_mas = (delta_obs - self.__delta0) / mastorad

        return delta_alpha_dx_mas, delta_delta_mas

class Satellite:

    def __init__(self, *args):
        self.init_parameters(*args)
        self.orbital_period = 365
        self.orbital_radius = 1.0

    def init_parameters(self, S=4.036, epsilon=np.radians(23.26), xi=np.radians(55), wz=120):
        """
        Sets satellite to initialization status.
        Args:
            S (float): -> dz/dlambda; change in z-axis of satellite with respect to solar longitudinal angle.
            epsilon (float): ecliptic angle [rad].
            xi (float): revolving angle [rad].
            wz (float): z component of inertial spin vector [arcsec/s].
        """
        self.S = S
        self.epsilon = epsilon
        self.xi = xi
        self.wz = wz * 60 * 60 * 24. * 0.0000048481368110954  # to [rad/day]
        self.ldot = 2 * np.pi / 365 # [rad/day]

    def ephemeris_bcrs(self, t):
        """
        Returns the barycentric ephemeris of the Gaia satellite, bg(t), at time t.
        :param t: float, time in days
        :return: 3d np.array [AU]
        """
        b_x_bcrs = self.orbital_radius*np.cos(2*np.pi/self.orbital_period*t)*np.cos(self.epsilon)
        b_y_bcrs = self.orbital_radius*np.sin(2*np.pi/self.orbital_period*t)*np.cos(self.epsilon)
        b_z_bcrs = self.orbital_radius*np.sin(2*np.pi/self.orbital_period*t)*np.sin(self.epsilon)

        bcrs_ephemeris_satellite = np.array([b_x_bcrs, b_y_bcrs, b_z_bcrs])

        return bcrs_ephemeris_satellite

class Attitude(Satellite):
    """
    Child class to Satellite.
    Calculates spline from attitude data of each our for 5 years by default.
    """
    def __init__(self, ti=0, tf=365*5, dt= 1/24.):
        Satellite.__init__(self)

        self.storage = []
        self.__init_state()
        self.__create_storage(ti, tf, dt)
        self.__init_state()
        self.__attitude_spline()

    def __init_state(self):
        """
        :return: initial status of satellite
        """
        self.t = 0
        self._lambda = 0
        self._beta = 0
        self.nu = 0
        self.omega = 0

        self.l, self.j, self.k = ft.ljk(self.epsilon)

        self.s = self.l*np.cos(self._lambda) + self.j*np.sin(self._lambda)

        self.attitude = self.__init_attitude()

        self.z = ft.lmn(self.attitude, np.array([0,0,1]))
        self.x = ft.lmn(self.attitude, np.array([1,0,0]))
        self.w = np.cross(np.array([0, 0, 1]), self.z)

    def __init_attitude(self):
        """
        :return: quaternion equivalent to initialization of satellite
        """
        q1 = Quaternion(np.cos(self.epsilon/2), np.sin(self.epsilon/2), 0, 0)
        q2 = Quaternion(np.cos(self._lambda/2), 0, 0, np.sin(self._lambda/2))
        q3 = Quaternion(np.cos((self.nu - (np.pi/2.))/2), np.sin((self.nu - (np.pi/2.)) / 2), 0, 0)
        q4 = Quaternion(np.cos((np.pi / 2. - self.xi)/2), 0, np.sin((np.pi/2. - self.xi)/2), 0)
        q5 = Quaternion(np.cos(self.omega/2.), 0, 0, np.sin(self.omega/2.))

        q_total = q1*q2*q3*q4*q5
        return q_total

    def __update(self, dt):
        """
        :param dt: time step to calculate derivatives of functions
        :return: update value of functions for next moment in time by calculating their infinitesimal change in dt
        """
        self.t = self.t + dt
        dL = self.ldot * dt
        self._lambda = self._lambda + dL

        nu_dot = (self.ldot/np.sin(self.xi))*(np.sqrt(self.S**2 - np.cos(self.nu)**2)
                                                + np.cos(self.xi)*np.sin(self.nu))
        d_nu = nu_dot * dt
        self.nu = self.nu + d_nu

        self.lamb_z = self._lambda + np.arctan2(np.tan(self.xi) * np.cos(self.nu), 1)
        self.beta_z = np.arcsin(np.sin(self.xi) * np.sin(self.nu))

        omega_dot = self.wz - nu_dot * np.cos(self.xi) - self.ldot * np.sin(self.xi) * np.sin(self.nu)
        d_omega = omega_dot * dt
        self.omega = self.omega + d_omega

        self.s = self.l * np.cos(self._lambda) + self.j * np.sin(self._lambda)

        z_dot = np.cross(self.k, self.z) * self.ldot + np.cross(self.s, self.z) * nu_dot
        dz = z_dot * dt
        self.z = self.z + dz
        self.z = self.z/np.linalg.linalg.norm(self.z)

        self.w = self.k * self.ldot + self.s * nu_dot + self.z * omega_dot

        #change attitude by deltaquat
        w_magnitude = np.linalg.norm(self.w)
        d_zheta = w_magnitude * dt
        delta_quat = ft.rotation_to_quat(self.w, d_zheta/2.) # w is not in bcrs frame.
        self.attitude = delta_quat * self.attitude

        # x axis rotates through quaternion multiplication
        x_quat = Quaternion(0, self.x[0], self.x[1], self.x[2])
        x_quat = delta_quat* x_quat* delta_quat.conjugate()
        self.x = x_quat.to_vector()

    def __attitude_spline(self):
        w_list = []
        x_list = []
        y_list = []
        z_list = []
        t_list = []
        for obj in self.storage:
            t_list.append(obj[0])
            w_list.append(obj[4].w)
            x_list.append(obj[4].x)
            y_list.append(obj[4].y)
            z_list.append(obj[4].z)

        self.s_x= interpolate.InterpolatedUnivariateSpline(t_list, x_list, k=4)
        self.s_y = interpolate.InterpolatedUnivariateSpline(t_list, y_list, k=4)
        self.s_z = interpolate.InterpolatedUnivariateSpline(t_list, z_list, k=4)
        self.s_w = interpolate.InterpolatedUnivariateSpline(t_list, w_list, k=4)

        self.func_attitude = lambda t: Quaternion(float(self.s_w(t)), float(self.s_x(t)), float(self.s_y(t)), float(self.s_z(t))).unit()
        self.func_x_axis_lmn  = lambda t: ft.xyz(self.func_attitude(t), self.x)

    def __reset_to_time(self, t, dt):
        '''
        Resets satellite to time t, along with all the parameters corresponding to that time.
        Args:
            t (float): time from J2000 [days]
        '''
        self.__init_state()
        n_steps = t / dt
        for i in np.arange(n_steps):
            self.__update(dt)

    def __create_storage(self, ti, tf, dt):
        '''
        Creates data necessary for step numerical methods performed in builtin method .update()
        Args:
            ti (float): integrating time lower limit [days]
            tf (float): integrating time upper limit [days]
            dt (float): step discretness of integration.
        Notes:
            stored in: attitude.storage
        '''

        if len(self.storage) == 0:
            self.__reset_to_time(ti, dt)

        n_steps = (tf - ti) / dt
        self.storage.append([self.t, self.w, self.z,
                             self.x, self.attitude, self.s])
        for i in np.arange(n_steps):
            self.__update(dt)
            self.storage.append([self.t, self.w, self.z,
                                 self.x, self.attitude, self.s])

        self.storage.sort(key=lambda x: x[0])

    def reset(self, ti, tf, dt):
        """
        :return: reset satellite to initialization status
        """
        self.storage.clear()
        self.__init_state()
        self.__create_storage(ti, tf, dt)
        self.__init_state()   #need to reset self.x to initial state for initialization of spline functions.
        self.__attitude_spline()

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

    def __init__(self, coarse_angle= np.radians(1)):
        self.coarse_angle = coarse_angle

        #create storage
        self.times_deep_scan = []
        self.obs_times = []

    def reset_memory(self):
        """
        :return: empty all attribute lists from scanner before beginning new scanning period.
        """
        self.obs_times.clear()
        self.times_deep_scan.clear()

    def coarse_scan(self, att, source, ti=0, tf=365*5, step=1/24):
        """
        Scans sky with a dot product technique to get rough times of observation.
        :return: None
        :action: self.times_deep_scan list filled with observation time windows.
        """
        self.reset_memory()
        for t in np.arange(ti, tf, step):
            to_star_unit = source.topocentric_function(att)(t)/np.linalg.norm(source.topocentric_function(att)(t))
            if np.arccos(np.dot(to_star_unit, att.func_x_axis_lmn(t))) < self.coarse_angle:
                self.times_deep_scan.append(t)

        print('Star crossing field of view %i times' %(len(self.times_deep_scan)))

    def fine_scan(self, att, source, step=1/24):

        def f(t):
            to_star_unit = source.topocentric_function(att)(t) / np.linalg.norm(source.topocentric_function(att)(t))
            star_syx_unit = ft.xyz(att.func_attitude(t), to_star_unit)
            diff_vector_xyz = star_syx_unit - ft.xyz(att.func_attitude(t), att.func_x_axis_lmn(t))
            return diff_vector_xyz


        for t in self.times_deep_scan:
            t_times= np.linspace(t - step, t + step, 1000)
            roots = self.root(f, t_times)
            if len(roots) != 0:
                self.obs_times.append(roots[0])

    def root(self, f, times_range):
        roots = []
        for idx, t in enumerate(times_range):
            if np.abs(f(t)[0]) < 0.01:
                if np.abs(f(t)[1]) < np.sin(np.radians(0.5)):
                    if np.abs(f(t)[2]) < np.sin(np.radians(1)):
                        roots.append(times_range[idx])
        return roots


def run():
    """
    :param days: number of days to run
    :param dt: time step
    :return: sky, scan, att
    """
    start_time = time.time()
    vega = Source("vega", 279.2333, 38.78, 128.91, 201.03, 286.23, -13.9)
    scan = Scanner()
    gaia = Attitude()
    scan.coarse_scan(gaia, vega)
    scan.fine_scan(gaia, vega)

    seconds = time.time() - start_time
    
    print('seconds:', seconds)
    return gaia, vega, scan
