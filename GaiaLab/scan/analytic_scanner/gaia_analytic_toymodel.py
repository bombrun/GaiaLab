# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:59:19 2018

@author: mdelvallevaro

"""

from . import frame_transformations as ft
from .quaternion import Quaternion
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

#vega_bcrs_au = ft.to_cartesian(np.radians(279.23), np.radians(38.78), 128.91)
#vega_bcrs_au_six = np.array([vega_bcrs_au[0],vega_bcrs_au[1], vega_bcrs_au[2], 0,0,0])

class Source:

    def __init__(self, name, alpha0, delta0, parallax, mu_alpha, mu_delta, mu_radial):
        """
        :param alpha0: deg
        :param delta0: deg
        :param parallax: mas
        :param mu_alpha: mas/yr
        :param mu_delta: mas/yr
        :param mu_radial: km/s
        """
        self.init_param(alpha0, delta0, parallax, mu_alpha, mu_delta, mu_radial)
        self.name = name
        self.alpha = self.__alpha0
        self.delta = self.__delta0

    def init_param(self, alpha0, delta0, parallax, mu_alpha, mu_delta, mu_radial):

        if type(alpha0) not in [int, float]:
            raise TypeError('alpha0 need to be int or float')
        if type(delta0) not in [int, float]:
            raise TypeError('delta0 need to be int or float')
        if type(parallax) not in [int, float]:
            raise TypeError('parallax need to be int or float')
        if type(mu_alpha) not in [int, float]:
            raise TypeError('mu_alpha need to be int or float')
        if type(mu_delta) not in [int, float]:
            raise TypeError('mu_delta need to be int or float')
        if type(mu_radial) not in [int, float]:
            raise TypeError('mu_radial need to be int or float')

        self.__alpha0= np.radians(alpha0)
        self.__delta0=  np.radians(delta0)
        self.parallax = parallax
        self.mu_alpha_dx = mu_alpha*np.cos(self.__delta0)
        self.mu_delta = mu_delta
        self.mu_radial = mu_radial


    def reset(self):
        """
        Reset star position to t=0
        """
        self.alpha = self.__alpha0
        self.delta = self.__delta0

    def set_time(self, t):
        """
        Sets star at position wrt bcrs at time t.
        :param t: [float][days]
        """
        if type(t) not in [float, int]:
            raise TypeError('t is not a float or int')
        if t<0:
            raise Warning('t is negative')

        mu_alpha_dx = self.mu_alpha_dx * 4.8473097e-9 / 365     #from mas/yr to rad/day
        mu_delta = self.mu_delta * 4.848136811095e-9 / 365      #from mas/yr to rad/day
        self.alpha = self.__alpha0 + mu_alpha_dx*t
        self.delta = self.__delta0 + mu_delta*t

    def barycentric_direction(self, t):
        """
        Direction unit vector to star from bcrs.
        :param t: [float][days]
        :return: ndarray 3D vector of [floats]
        """
        self.set_time(t)
        u_bcrs_direction = ft.to_direction(self.alpha, self.delta)
        return u_bcrs_direction #no units, just a unit direction

    def barycentric_coor(self, t):
        """
        Vector to star wrt bcrs-frame.
        alpha: [float][rad]
        delta: [float][rad]
        parallax: [float][rad]
        :param t: [float][days]
        :return: ndarray, length 3, components [floats][parsecs]
        """
        self.set_time(t)
        u_bcrs = ft.to_cartesian(self.alpha, self.delta, self.parallax)
        return u_bcrs

    def topocentric_function(self, satellite):
        """
        :param satellite: satellite [class object]
        :return: [lambda function] of the position of the star from the satellite's lmn frame.
        """
        if isinstance(satellite, Satellite) != True:
            raise TypeError('arg is not Satellite object')

        p, q, r = ft.pqr(self.alpha, self.delta)
        mastorad = 2*np.pi/(1000*360*3600)
        kmtopc = 3.24078e-14
        sectoday = 3600*24
        AUtopc = 4.8481705933824e-6

        mu_alpha_dx = self.mu_alpha_dx*mastorad/365   #mas/yr to rad/day
        mu_delta = self.mu_delta*mastorad/365  #mas/yr to rad/day
        mu_radial = self.parallax*mastorad*self.mu_radial*kmtopc*sectoday #km/s to aproximation rad/day

        topocentric_function = lambda t: self.barycentric_coor(0) + t*(p*mu_alpha_dx+ q*mu_delta + r*mu_radial) \
                                         - satellite.ephemeris_bcrs(t)*AUtopc
        return topocentric_function

    def topocentric_angles(self, satellite, t):
        """
        Calculates the angles of movement of the star from bcrs.
        :param satellite: satellite object
        :param t: [days]
        :return: alpha, delta, delta alpha, delta delta [mas]
        """
        mastorad = 2 * np.pi / (1000 * 360 * 3600)
        u_lmn = self.topocentric_function(satellite)(t)
        u_lmn_unit = u_lmn/np.linalg.norm(u_lmn)
        alpha_obs, delta_obs, radius = ft.to_polar(u_lmn_unit)

        if alpha_obs < 0:
            alpha_obs = (alpha_obs + 2*np.pi)/mastorad

        delta_alpha_dx_mas = (alpha_obs - self.__alpha0) * np.cos(self.__delta0) / mastorad
        delta_delta_mas = (delta_obs - self.__delta0) / mastorad

        return alpha_obs, delta_obs, delta_alpha_dx_mas, delta_delta_mas    # mas

class Satellite:
    """
    Class Object, parent to Attitude, that represents Gaia.

    %run: epsilon = np.radians(23.26), xi = np.radians(55).

    :param S: change in the z-axis of satellite wrt solar longitudinal angle. [float]
    :param epsilon: ecliptical angle [rad]
    :param xi: revolving angle [rad]
    :param wz: z component of inertial spin vector [arcsec/s]
    :action: Sets satellite to initialization status.
    """
    def __init__(self, *args):
        """
        :orbital_period: [days]
        :orbital_radius: [AU]
        """
        self.init_parameters(*args)
        self.orbital_period = 365
        self.orbital_radius = 1.0

    def init_parameters(self, S=4.036, epsilon=np.radians(23.26), xi=np.radians(55), wz=120):
        self.S = S
        self.epsilon = epsilon
        self.xi = xi
        self.wz = wz * 60 * 60 * 24. * 0.0000048481368110954  # to [rad/day]
        self.ldot = 2 * np.pi / 365 # [rad/day]

    def ephemeris_bcrs(self, t):
        """
        Returns the barycentric ephemeris of the Gaia satellite at time t.
        :param t: float [days]
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
    Creates spline from attitude data for 5 years by default.
    :param ti: initial time, float [day]
    :param tf: final time, float [day]
    :param dt: time step for creation of discrete data fed to spline, float [day].
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

        self.z = ft.to_lmn(self.attitude, np.array([0,0,1]))
        self.x = ft.to_lmn(self.attitude, np.array([1,0,0]))
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
        """
        Creates spline for each component of the attitude quaternion:
            s_x, s_y, s_z, s_w
        Attributes
        -----------
        :func_attitude: lambda func, returns: attitude quaternion at time t from spline.
        :func_x_axis_lmn: lambda func, returns: position of x_axis of satellite at time t, in lmn frame.
        :func_z_axis_lmn: lambda func, returns: position of z_axis of satellite at time t, in lmn frame.
        """
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
        self.func_x_axis_lmn  = lambda t: ft.to_lmn(self.func_attitude(t), np.array([1,0,0]))
        self.func_z_axis_lmn = lambda t: ft.to_lmn(self.func_attitude(t), np.array([0,0,1]))


    def __reset_to_time(self, t, dt):
        '''
        Resets satellite to time t, along with all the parameters corresponding to that time.
        :param t: to time [day]
        :param dt: [day]
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

    def __init__(self, wide_angle = np.radians(20),  scan_line_height = np.radians(5)):
        """
        :param wide_angle: angle for first dot-product-rough scan of all the sky.
        :param scan_line_height: condition for the line_height of the scanner (z-axis height in lmn)

        Attributes
        -----------
        :coarse_angle: uses scan_line_height to get measurements after wide_angle dot product. [rad]
        :times_wide_scan: times where star in wide_angle field of view. [days]
        :times_coarse_scan: times where star in wide_angle and coarse angle field of view [days]
        :optimize_roots: [func] minimized solutions
        :roots: [func] found roots
        :obs_times: the times of the roots (i.e. self.roots.x) where the star precisely crosses the line of view [days]
        """
        self.wide_angle = wide_angle
        self.scan_line_height = scan_line_height/2.
        self.coarse_angle = self.scan_line_height

        self.z_threshold = np.sin(self.scan_line_height)

        self.times_wide_scan = []
        self.times_coarse_scan = []

        self.times_optimize = []
        self.optimize_roots = []
        self.roots = []
        self.obs_times = []

    def reset_memory(self):
        """
        :return: empty all attribute lists from scanner before beginning new scanning period.
        """
        self.times_wide_scan.clear()
        self.times_coarse_scan.clear()

        self.times_optimize.clear()
        self.optimize_roots.clear()
        self.roots.clear()
        self.obs_times.clear()

    def start(self, att, source, ti=0, tf=365 * 5):
        self.wide_coarse_double_scan(att, source, ti, tf)
        self.fine_scan(att, source)

    def wide_coarse_double_scan(self, att, source, ti=0, tf=365 * 5):
        """
        Scans sky with a dot product technique to get rough times of observation.
        :action: self.times_deep_scan list filled with observation time windows.
        """
        if isinstance(att, Attitude) != True:
            return TypeError('firs argument is not an Attitude object')
        if isinstance(source, Source) != True:
            return TypeError('second argument is not a Source object')

        self.reset_memory()
        step_wide = self.wide_angle / (2 * np.pi * 4)
        for t in np.arange(ti, tf, step_wide):
            to_star_unit = source.topocentric_function(att)(t) / np.linalg.norm(source.topocentric_function(att)(t))
            if np.arccos(np.dot(to_star_unit, att.func_x_axis_lmn(t))) < self.wide_angle:
                self.times_wide_scan.append(t)

        step_coarse = self.coarse_angle / (2 * np.pi * 4)
        for t_wide in self.times_wide_scan:
            for t in np.arange(t_wide - step_wide / 2, t_wide + step_wide / 2, step_coarse):
                to_star_unit = source.topocentric_function(att)(t) / np.linalg.norm(source.topocentric_function(att)(t))
                if np.arccos(np.dot(to_star_unit, att.func_x_axis_lmn(t))) < self.coarse_angle:
                    self.times_coarse_scan.append(t)


    def fine_scan(self, att, source):

        def phi_objective(t):
            t = float(t)
            to_star_unit_lmn = source.topocentric_function(att)(t) / np.linalg.norm(source.topocentric_function(att)(t))
            phi_vector_lmn = to_star_unit_lmn - att.func_x_axis_lmn(t)
            phi_vector_xyz = ft.to_xyz(att.func_attitude(t), phi_vector_lmn)
            return np.abs(phi_vector_xyz[1])

        def z_condition(t):
            t = float(t)
            to_star_unit_lmn = source.topocentric_function(att)(t) / np.linalg.norm(source.topocentric_function(att)(t))
            diff_vector = to_star_unit_lmn - att.func_x_axis_lmn(t)
            diff_vector_xyz = ft.to_xyz(att.func_attitude(t), diff_vector)
            z_threshold = np.sin(self.scan_line_height)
            return z_threshold - np.abs(diff_vector_xyz[2])

        con1 = {'type': 'ineq', 'fun': z_condition}

        time_step = self.coarse_angle / (2 * np.pi * 4)

        #find times where possible solutions are
        for i in self.times_coarse_scan:
            def t_condition(t):
                if i - time_step < t < i + time_step:
                    return 1.0
                else:
                    return -1.0

            con2 = {'type': 'ineq', 'fun': t_condition}

            optimize_root = optimize.minimize(phi_objective, i, method='COBYLA', constraints=[con1, con2])
            if optimize_root.success == True:
                self.times_optimize.append(float(optimize_root.x))
                self.optimize_roots.append(optimize_root)

        #find roots for phi
        for obj in self.optimize_roots:
            root = optimize.root(phi_objective, [obj.x])
            self.roots.append(root)
            self.obs_times.append(float(root.x))

        #remove duplicates
        self.obs_times = list(set(self.obs_times))
        self.obs_times.sort()

def phi(source, att, t):
    """
    Calculates the diference between the x-axis of the satellite and the direction vector to the star.
    Once this is calculated, it checks how far away is in the alpha direction (i.e. the y-component) wrt IRS.
    :param source: Source [object]
    :param att: Attitude [object]
    :param t: time [float][days]
    :return: [float] angle, alpha wrt IRS.
    """
    t = float(t)
    u_lmn_unit = source.topocentric_function(att)(t) / np.linalg.norm(source.topocentric_function(att)(t))
    phi_value_lmn = u_lmn_unit - att.func_x_axis_lmn(t)
    phi_value_xyz = ft.to_xyz(att.func_attitude(t), phi_value_lmn)
    return np.arcsin(phi_value_xyz[1]), np.arcsin(phi_value_xyz[2])

def run():
    """
    :param days: number of days to run
    :param dt: time step
    :return: sky, scan, att
    """
    start_time = time.time()
    sirio = Source("sirio", 101.28, -16.7161, 379.21, -546.05, -1223.14, -7.6)
    vega = Source("vega", 279.2333, 38.78, 128.91, 201.03, 286.23, -13.9)
    proxima = Source("proxima",217.42, -62, 768.7, 3775.40, 769.33, 21.7)

    scanSirio = Scanner(np.radians(20), np.radians(2))
    scanVega =  Scanner(np.radians(20), np.radians(2))
    scanProxima = Scanner(np.radians(20), np.radians(2))
    gaia = Attitude()
    print (time.time() - start_time)

    scanSirio.start(gaia, sirio)
    scanVega.start(gaia, vega)
    scanProxima.start(gaia, proxima)
    print (time.time() - start_time)

    seconds = time.time() - start_time
    print('Total seconds:', seconds)
    return gaia, sirio, scanSirio, vega, scanVega, proxima, scanProxima

