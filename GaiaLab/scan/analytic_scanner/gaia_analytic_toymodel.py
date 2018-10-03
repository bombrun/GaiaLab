# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:59:19 2018

@author: mdelvallevaro

modified by: LucaZampieri 2018

Contain the classes:
- Source
- Scanner
- Satellite
- Attitude <-- child of Satellite

This work has been inspired by what has been found in the following notes and
papers:
- (Lindegren, SAG-LL-14)
- (Lindegren, SAG-LL-30)
- (Lindegren, SAG-LL-35)
- The astrometric core solution for the gaia mission, overview of models, algorithms,
and software implementation, L.Lindegren et al.


"""

import frame_transformations as ft
from quaternion import Quaternion
import constants as const

import numpy as np
import time
from scipy import interpolate
from scipy import optimize

from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# vega = Source("vega", 279.2333, 38.78, 128.91, 201.03, 286.23, -13.9)
# proxima = Source("proxima",217.42, -62, 768.7, 3775.40, 769.33, 21.7)
# sirio = Source("sirio", 101.28, -16.7161, 379.21, -546.05, -1223.14, -7.6)

# vega_bcrs_au = ft.adp_to_cartesian(np.radians(279.23), np.radians(38.78), 128.91)
# vega_bcrs_au_six = np.array([vega_bcrs_au[0],vega_bcrs_au[1], vega_bcrs_au[2], 0,0,0])


class Source:
    """
    Source class implemented to represent a source object in the sky
    """

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

        self.__alpha0 = np.radians(alpha0)
        self.__delta0 = np.radians(delta0)
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
            raise TypeError('t is not a float or int, but instead of type %r.' % type(t))
        if t < 0:
            raise Warning('t is negative')

        mu_alpha_dx = self.mu_alpha_dx * 4.8473097e-9 / 365     # from mas/yr to rad/day
        mu_delta = self.mu_delta * 4.848136811095e-9 / 365      # from mas/yr to rad/day
        # mu_alpha_dx = self.mu_alpha_dx * const.rad_per_mas / const.days_per_year     # from mas/yr to rad/day
        # mu_delta = self.mu_delta * const.rad_per_mas / const.days_per_year           # from mas/yr to rad/day

        self.alpha = self.__alpha0 + mu_alpha_dx*t
        self.delta = self.__delta0 + mu_delta*t

    def barycentric_direction(self, t):
        """
        Direction unit vector to star from bcrs.
        :param t: [float][days]
        :return: ndarray 3D vector of [floats]
        """
        self.set_time(t)
        u_bcrs_direction = ft.polar_to_direction(self.alpha, self.delta)
        return u_bcrs_direction  # no units, just a unit direction

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
        u_bcrs = ft.adp_to_cartesian(self.alpha, self.delta, self.parallax)
        return u_bcrs

    def unit_topocentric_function(self, satellite, t):
        """
        Compute the topocentric_function direction
        The horizontal coordinate system, also known as topocentric coordinate
        system, is a celestial coordinate system that uses the observer's local
        horizon as the fundamental plane. Coordinates of an object in the sky are
        expressed in terms of altitude (or elevation) angle and azimuth.
        :param satellite: satellite [class object]
        :return: [array] (x,y,z) direction-vector of the star from the satellite's lmn frame.
        """
        # if not isinstance(satellite, Satellite):
        #     raise TypeError('Expected Satellite, but got {} instead'.format(type(satellite)))

        p, q, r = ft.compute_pqr(self.alpha, self.delta)

        mu_alpha_dx = self.mu_alpha_dx * const.rad_per_mas / const.days_per_year   # mas/yr to rad/day
        mu_delta = self.mu_delta * const.rad_per_mas / const.days_per_year  # mas/yr to rad/day
        # km/s to aproximation rad/day
        mu_radial = self.parallax * const.rad_per_mas * self.mu_radial * const.km_per_pc * const.sec_per_day

        # topocentric_function direction
        topocentric = self.barycentric_coor(0) + t*(p*mu_alpha_dx + q * mu_delta + r*mu_radial) \
            - satellite.ephemeris_bcrs(t) * const.AU_per_pc
        norm_topocentric = np.linalg.norm(topocentric)

        return topocentric / norm_topocentric

    def topocentric_angles(self, satellite, t):
        """
        Calculates the angles of movement of the star from bcrs.
        :param satellite: satellite object
        :param t: [days]
        :return: alpha, delta, delta alpha, delta delta [mas]
        """
        # mastorad = 2 * np.pi / (1000 * 360 * 3600)

        u_lmn_unit = self.unit_topocentric_function(satellite, t)
        alpha_obs, delta_obs, radius = ft.vector_to_polar(u_lmn_unit)

        if alpha_obs < 0:
            alpha_obs = (alpha_obs + 2*np.pi) / const.rad_per_mas

        delta_alpha_dx_mas = (alpha_obs - self.__alpha0) * np.cos(self.__delta0) / const.rad_per_mas
        delta_delta_mas = (delta_obs - self.__delta0) / const.rad_per_mas

        return alpha_obs, delta_obs, delta_alpha_dx_mas, delta_delta_mas  # mas


class Satellite:
    """
    Class Object, parent to Attitude, that represents Gaia.

    %run: epsilon = np.radians(23.26), xi = np.radians(55).

    :param S: change in the z-axis of satellite wrt solar longitudinal angle.
     [float]
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
        self.orbital_period = const.days_per_year
        self.orbital_radius = 1.0

    def init_parameters(self, S=const.S, epsilon=np.radians(const.epsilon), xi=np.radians(const.xi), wz=const.w_z):
        self.S = S

        # obliquity of equator. This is a constant chosen to be 23º 26' 21.448''
        self.epsilon = epsilon

        # "ksi" revolving angle. At any time the z axis is at this constant angle from s⃗ s→.
        # For Gaia, the current choice is 55º.
        self.xi = xi
        self.wz = wz * const.sec_per_day * const.AU_per_pc  # to [rad/day]
        # self.wz = wz * 60 * 60 * 24. * 0.0000048481368110954  # to [rad/day]

        # Nominal longitud of the sun in the ecliptic plane
        self.lambda_dot = 2 * np.pi / const.days_per_year  # [rad/day] (lambda dot set as const)
        # self.ldot = 2 * np.pi / const.days_per_year  # [rad/day] (lambda dot set as const)

    def ephemeris_bcrs(self, t):
        """
        Defines the orbit of the satellite around the sun
        Returns the barycentric ephemeris of the Gaia satellite at time t.
        Equivalently written b_G(t)

        :param t: float [days]
        :return: 3D np.array [AU]
        """
        # Assuming it is a circle tilted by epsilon:
        b_x_bcrs = self.orbital_radius*np.cos(2*np.pi/self.orbital_period*t)*np.cos(self.epsilon)
        b_y_bcrs = self.orbital_radius*np.sin(2*np.pi/self.orbital_period*t)*np.cos(self.epsilon)
        b_z_bcrs = self.orbital_radius*np.sin(2*np.pi/self.orbital_period*t)*np.sin(self.epsilon)

        bcrs_ephemeris_satellite = np.array([b_x_bcrs, b_y_bcrs, b_z_bcrs])
        return bcrs_ephemeris_satellite


class Attitude(Satellite):
    """
    Child class to Satellite.
    Creates spline from attitude data for 5 years by default.
    info: (see e.g. Lindegren, SAG-LL-35)
        The Nominal Scanning Law (NSL) for gaia is descibed by two constant
        angles:
        - Epsilon: obliquity of equator
        - Xi (greek letter): revolving angles
        and 3 angles that increase continuously but non-uniformly:
        - _lambda(t): nominal longitude of the sun
        - nu(t): revolving phase
        - omega(t): spin phase
        With also initial values nu(0), omega(0) at time t_0 the NSL is completely
        specified.

    :param ti: initial time, float [day]
    :param tf: final time, float [day]
    :param dt: time step for creation of discrete data fed to spline, float [day].
    """
    def __init__(self, ti=0, tf=5*const.days_per_year, dt=1/24., *args):
        Satellite.__init__(self, *args)
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

        self.l, self.j, self.k = ft.compute_ljk(self.epsilon)

        self.s = self.l*np.cos(self._lambda) + self.j*np.sin(self._lambda)

        self.attitude = self.__init_attitude()

        self.z = ft.xyz_to_lmn(self.attitude, np.array([0, 0, 1]))
        self.x = ft.xyz_to_lmn(self.attitude, np.array([1, 0, 0]))
        self.w = np.cross(np.array([0, 0, 1]), self.z)

    def __init_attitude(self):
        """
        (Lindegren, SAG-LL-35, Eq.6)
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
        Update value of functions for next moment in time by calculating their infinitesimal change in dt
        :param dt: time step to calculate derivatives of functions
        """

        self.t = self.t + dt

        # update lambda
        self._lambda = self._lambda + self.lambda_dot * dt  # Updates lambda

        # Update nu
        nu_dot = (self.lambda_dot/np.sin(self.xi))*(np.sqrt(self.S**2 - np.cos(self.nu)**2)
                                                    + np.cos(self.xi)*np.sin(self.nu))
        self.nu = self.nu + nu_dot * dt

        # how does the longitude and latitude of the z axis changes with time:
        self.lamb_z = self._lambda + np.arctan2(np.tan(self.xi) * np.cos(self.nu), 1)
        self.beta_z = np.arcsin(np.sin(self.xi) * np.sin(self.nu))

        # Update Omega
        omega_dot = self.wz - nu_dot * np.cos(self.xi) - self.lambda_dot * np.sin(self.xi) * np.sin(self.nu)
        self.omega = self.omega + omega_dot * dt

        # Update S
        self.s = self.l * np.cos(self._lambda) + self.j * np.sin(self._lambda)

        # Update z
        z_dot = np.cross(self.k, self.z) * self.lambda_dot + np.cross(self.s, self.z) * nu_dot
        self.z = self.z + z_dot * dt
        self.z = self.z/np.linalg.linalg.norm(self.z)

        # Update w (total inertial rotation of the telescope)
        self.w = self.k * self.lambda_dot + self.s * nu_dot + self.z * omega_dot

        # change attitude by delta_quat
        w_magnitude = np.linalg.norm(self.w)
        d_zheta = w_magnitude * dt
        delta_quat = ft.rotation_to_quat(self.w, d_zheta/2.)  # w is not in bcrs frame.
        self.attitude = delta_quat * self.attitude

        # x axis rotates through quaternion multiplication
        x_quat = Quaternion(0, self.x[0], self.x[1], self.x[2])
        x_quat = delta_quat * x_quat * delta_quat.conjugate()
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

        # Splines for each coordinates i, i_list at each time in t_list of order k
        self.s_x = interpolate.InterpolatedUnivariateSpline(t_list, x_list, k=4)
        self.s_y = interpolate.InterpolatedUnivariateSpline(t_list, y_list, k=4)
        self.s_z = interpolate.InterpolatedUnivariateSpline(t_list, z_list, k=4)
        self.s_w = interpolate.InterpolatedUnivariateSpline(t_list, w_list, k=4)

        # Attitude
        self.func_attitude = lambda t: Quaternion(float(self.s_w(t)), float(self.s_x(t)), float(self.s_y(t)),
                                                  float(self.s_z(t))).unit()
        # Attitude in the lmn frame
        self.func_x_axis_lmn = lambda t: ft.xyz_to_lmn(self.func_attitude(t), np.array([1, 0, 0]))
        self.func_y_axis_lmn = lambda t: ft.xyz_to_lmn(self.func_attitude(t), np.array([0, 1, 0]))
        self.func_z_axis_lmn = lambda t: ft.xyz_to_lmn(self.func_attitude(t), np.array([0, 0, 1]))

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
        Creates data necessary for step numerical methods performed in builtin method .__update()
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
        self.__init_state()   # need to reset self.x to initial state for initialization of spline functions.
        self.__attitude_spline()


class Scanner:

    def __init__(self, wide_angle=np.radians(20),  scan_line_height=np.radians(5)):
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
        self.wide_angle = wide_angle  # Threshold for the x-axis
        self.scan_line_height = scan_line_height/2.  # Threshold over z axis
        self.coarse_angle = self.scan_line_height  # angle of view of the coarse scan

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

    def start(self, att, source, ti=0, tf=5*const.days_per_year):
        print('Starting wide_scan with time from {} to {} days'.format(ti, tf))
        self.wide_scan(att, source, ti, tf)
        print('Finished wide_scan!')

        print('Starting coarse_scan with time from {} to {} days'.format(ti, tf))
        self.coarse_scan(att, source, ti, tf)
        print('Finished coarse_scan!')

        print('Starting fine_scan:')
        self.fine_scan(att, source)
        print('Finished fine_scan!')

    def wide_scan(self, att, source, ti=0, tf=5*const.days_per_year):
        """
        Scans sky with a dot product technique to get rough times of observation.
        :action: self.times_wide_scan list filled with observation time windows.
        """
        # if not isinstance(att, Attitude):
        #    raise TypeError('Expected Attitude, but got {} instead'.format(type(att)))
        # if not isinstance(source, Source):
        #    raise TypeError('Expected Source, but got {} instead'.format(type(source)))

        # Reset the memory of the previous scans
        self.reset_memory()

        t_0 = time.time()  # t0 of the timer

        self.step_wide = self.wide_angle / (2 * np.pi * 4)
        for t in np.arange(ti, tf, self.step_wide):
            to_star_unit = source.unit_topocentric_function(att, t)
            angle_source_xaxis = np.arccos(np.dot(to_star_unit, att.func_x_axis_lmn(t)))
            if angle_source_xaxis < self.wide_angle:
                self.times_wide_scan.append(t)
        time_wide = time.time()  # time after wide scan
        print('wide scan lasted {} seconds'.format(time_wide - t_0))
        print('Found {} times with wide scan'.format(len(self.times_wide_scan)))

        # # Alternative way to do it:
        # my_ts = np.arange(ti, tf, step_wide)
        # def f(x):
        #     to_star_unit = source.unit_topocentric_function(att, x)
        #     return np.arccos(np.dot(to_star_unit, att.func_x_axis_lmn(x))) < self.wide_angle
        # array_map = np.array(list(map(f, my_ts)))
        # self.times_wide_scan = list(my_ts[np.nonzero(array_map)])
        # #

    def coarse_scan(self, att, source, ti=0, tf=5*const.days_per_year):
        t_0 = time.time()  # reset the  t_0 of the time
        # Make the coarse angle scan
        step_coarse = self.coarse_angle / (2 * np.pi * 4)
        for t_wide in self.times_wide_scan:
            for t in np.arange(t_wide - self.step_wide / 2, t_wide + self.step_wide / 2, step_coarse):
                to_star_unit = source.unit_topocentric_function(att, t)
                if np.arccos(np.dot(to_star_unit, att.func_x_axis_lmn(t))) < self.coarse_angle:
                    self.times_coarse_scan.append(t)
        time_coarse = time.time()  # time after coarse scan
        print('Coarse scan lasted {} seconds'.format(time_coarse - t_0))
        print('Found {} times with coarse scan'.format(len(self.times_coarse_scan)))

    # fine_scan function
    def fine_scan(self, att, source, tolerance=1e-3):
        """
        Find the exact time in which the source is seen. Only the times when the
        source is in the field of view are scanned, i.e. self.times_coarse_scan.
        :param att: [Attitude object]
        :param source: [Source object]
        :param tolerance: [int,float, optional] [days] tolerance up to which we distinguish
            two observations
        :action: Find the observation time of the sources
        """

        def phi_objective(t):
            t = float(t)
            u_lmn_unit = source.unit_topocentric_function(att, t)
            phi_vector_lmn = u_lmn_unit - att.func_x_axis_lmn(t)
            phi_vector_xyz = ft.lmn_to_xyz(att.func_attitude(t), phi_vector_lmn)
            return np.abs(phi_vector_xyz[1])

        def z_condition(t):
            t = float(t)
            u_lmn_unit = source.unit_topocentric_function(att, t)
            phi_vector_lmn = u_lmn_unit - att.func_x_axis_lmn(t)
            phi_vector_xyz = ft.lmn_to_xyz(att.func_attitude(t), phi_vector_lmn)
            z_threshold = np.sin(self.scan_line_height)
            return z_threshold - np.abs(phi_vector_xyz[2])  # >= 0 for scipy.optimize.minimize

        con1 = {'type': 'ineq', 'fun': z_condition}  # inequality constraint: z_condition >= 0

        time_step = self.coarse_angle / (2 * np.pi * 4)
        print('time_step: {}'.format(time_step))

        t_0 = time.time()  # set t_0 of the timer
        # find times where possible solutions are
        for i in self.times_coarse_scan:
            def t_condition(t):
                if i - time_step < t < i + time_step:
                    return 1.0
                else:
                    return -1.0

            con2 = {'type': 'ineq', 'fun': t_condition}

            optimize_root = optimize.minimize(phi_objective, i, method='COBYLA', constraints=[con1, con2])
            if optimize_root.success:
                self.times_optimize.append(float(optimize_root.x))
                self.optimize_roots.append(optimize_root)
        time_optimize_root = time.time()  # time after wide scan
        print('phi_minimization lasted {} seconds'.format(time_optimize_root - t_0))

        t_0 = time.time()  # reset t_0 of the timer
        # find roots for phi
        for obj in self.optimize_roots:
            root = optimize.root(phi_objective, [obj.x])
            self.roots.append(root)
            self.obs_times.append(float(root.x))
        time_phi_root = time.time()  # time after wide scan
        print('wide scan lasted {} seconds'.format(time_phi_root - t_0))

        # remove identical duplicates
        print('original obs_times: {}'.format(self.obs_times))
        self.obs_times = list(set(self.obs_times))
        self.obs_times.sort()  # to leave them in increasing order
        print('identical duplicates removal obs_time: {}'.format(self.obs_times))


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
    u_lmn_unit = source.unit_topocentric_function(att, t)
    phi_value_lmn = u_lmn_unit - att.func_x_axis_lmn(t)
    phi_value_xyz = ft.lmn_to_xyz(att.func_attitude(t), phi_value_lmn)
    phi = np.arcsin(phi_value_xyz[1])
    eta = np.arcsin(phi_value_xyz[2])
    return phi, eta


def run():
    """
    Create the objects source for Sirio, Vega and Proxima as well
    as the corresponding scanners and the attitude object of Gaia.
    Then scan the sources from Gaia and print the time.
    :return: gaia, sirio, scanSirio, vega, scanVega, proxima, scanProxima
    """
    start_time = time.time()
    sirio = Source("sirio", 101.28, -16.7161, 379.21, -546.05, -1223.14, -7.6)
    vega = Source("vega", 279.2333, 38.78, 128.91, 201.03, 286.23, -13.9)
    proxima = Source("proxima", 217.42, -62, 768.7, 3775.40, 769.33, 21.7)

    scanSirio = Scanner(np.radians(20), np.radians(2))
    scanVega = Scanner(np.radians(20), np.radians(2))
    scanProxima = Scanner(np.radians(20), np.radians(2))
    gaia = Attitude()
    print(time.time() - start_time)

    scanSirio.start(gaia, sirio)
    scanVega.start(gaia, vega)
    scanProxima.start(gaia, proxima)
    print(time.time() - start_time)

    seconds = time.time() - start_time
    print('Total seconds:', seconds)
    return gaia, sirio, scanSirio, vega, scanVega, proxima, scanProxima


################################################################################
# # isInstance functions
# this function should not be used
def test_is_attitude(other):
    """ Tests if (other) is of type attitude. Raise exception otherwise.
    """
    if not isinstance(other, Attitude):
        raise TypeError('{} is not an Attitude object'.format(type(other)))
    else:
        pass


# not used yet
def test_object_type(other, type_str):
    """
    Tests if (other) is of type (type_str). Raise exception otherwise.
    :param other: Variable which type should be tested
    :param type_str: [str] string containing the object type we want to test.
    """

    possible_types = {"Source": Source,
                      "Satellite": Satellite,
                      "Attitude": Attitude,
                      "Scanner": Scanner}
    if type_str not in possible_types:
        raise TypeError('Expected type "{}" is not part of the possible_types'.format(type_str))

    expected_type = possible_types[type_str]

    if not isinstance(other, expected_type):
        raise TypeError('Type "{}" is not "{}"'.format(type(other), type_str))
