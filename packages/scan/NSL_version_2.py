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
from mpl_toolkits.mplot3d import Axes3D

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
        for i in range(n):
            alpha = np.random.uniform(0, 2 * np.pi)
            delta = np.random.uniform(-np.pi / 2, np.pi / 2)
            solar_radius = np.random.uniform(1, 5)
            parallax = np.arctan2(1, solar_radius)
            self.elements.append(Source(alpha, delta, parallax))


class Source:
    """
    Defines source star by horizontal coordinate system parameters.
    Args:
        alpha (float): galactic longitudinal angle [rad].
        delta (float): galactic altitude angle (rad),

    Attributes:
        coor (np.dnarray): (alpha, delta) horizontal coordinate system.
    """

    def __init__(self, alpha0, delta0, parallax, mu_alpha, mu_delta, mu_radial):

        self.init_param(alpha0, delta0, parallax, mu_alpha, mu_delta, mu_radial)
        self.alpha = self.alpha0
        self.delta = self.delta0

    def init_param(self, alpha0, delta0, parallax, mu_alpha, mu_delta, mu_radial):
        self.alpha0= alpha0
        self.delta0= delta0
        self.parallax = parallax
        self.mu_alpha_dx = mu_alpha*np.cos(delta0)
        self.mu_delta = mu_delta
        self.mu_radial = mu_radial

    def reset(self):
        self.alpha = self.alpha0
        self.delta = self.delta0

    def update(self, t): #implement here relativistic effects
        self.alpha = self.alpha0 + self.mu_alpha_dx*t
        self.delta = self.delta0 + self.mu_delta*t

    def coor_bcrs(self, t):
        self.update(t)
        u_bcrs = ft.cartesian_coord(self.alpha, self.delta, self.parallax)
        return u_bcrs

    def coor_xyz(self, satellite, t):
        self.update(t)
        p, q, r = ft.pqr(self.alpha, self.delta)
        u = self.coor_bcrs(t) + t*(p*self.mu_alpha_dx + q*self.mu_delta + r*self.mu_radial)
        u_xyz = u - (self.parallax/satellite.orbital_radius) * satellite.ephemeris_bcrs(t)
        return u_xyz


class Satellite:
    def __init__(self, *args):
        self.init_parameters(*args)

    def init_parameters(self, S=4.036, epsilon=np.radians(23.26), xi=np.radians(45), wz=120):
        """
        Sets satellite to initialization status.
        Args:
            S (float): -> dz/dlambda; change in z-axis of satellite with respect to solar longitudinal angle.
            epsilon (float): ecliptical angle [rad].
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
        :return: 3d np.array
        """
        self.orbital_period = 365
        self.orbital_radius = 1.0

        b_x_bcrs = self.orbital_radius*np.cos(2*np.pi/self.orbital_period*t)
        b_y_bcrs = self.orbital_radius*np.sin(2*np.pi/self.orbital_period*t)
        b_z_bcrs = t*0

        bcrs_ephemeris_satellite = np.array([b_x_bcrs, b_y_bcrs, b_z_bcrs])

        return bcrs_ephemeris_satellite

    def srs_to_bcrs(self, vector_srs, t):
        """
        Change frame of reference from a vector in lmn to barycentric frame.
        :param vector_srs: 3d vector in SRS frame
        :param t: float, time in days
        :return: 3d np.array
        """
        l = np.array([1, 0, 0])
        j = np.array([0, np.cos(self.epsilon), -np.sin(self.epsilon)])
        k = np.array([0, np.sin(self.epsilon), np.cos(self.epsilon)])

        A = np.vstack([l, j, k])
        A_matrix = A.reshape(3, 3)

        vector_bcrs = np.dot(A_matrix, vector_srs) + self.ephemeris_bcrs(t)
        return vector_bcrs


class Attitude(Satellite):
    """
    Child class to Satellite.
    """
    def __init__(self):
        Satellite.__init__(self)
        self.init_state()
        self.storage = []

    def init_state(self):
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

        self.attitude = self.init_attitude()

        self.z = (self.attitude * Quaternion(0,0,0,1)*self.attitude.conjugate()).to_vector()
        self.x = (self.attitude * Quaternion(0,1,0,0)*self.attitude.conjugate()).to_vector()
        self.w = np.cross(np.array([0, 0, 1]), self.z)

    def init_attitude(self):
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

    def reset(self):
        """
        :return: reset satellite to initialization status
        """
        self.init_state()
        self.storage.clear()

    def update(self, dt):
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

    def long_reset_to_time(self, t, dt):
        # this is slowing down create_storage but it is very exact.
        # Another way to do it would be if functions in attitude.update where analytical.
        '''
        Resets satellite to time t, along with all the parameters corresponding to that time.
        Args:
            t (float): time from J2000 [days]
        '''
        self.init_state()
        n_steps = t/dt
        for i in np.arange(n_steps):
            self.update(dt)

    def short_reset_to_time(self, t):

        temp_list = [obj for obj in self.storage if obj[0] <= t]
        list_element = temp_list[0]
        self.t = list_element[0]
        self.w = list_element[1]
        self.z = list_element[2]
        self.x = list_element[3]
        self.attitude = list_element[4]
        self.lamb_z = list_element[5]
        self.beta_z = list_element[6]
        self.s = list_element[7]

    def create_storage(self, ti, tf, dt):
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
            ti = ti
        if len(self.storage) > 0:
            self.short_reset_to_time(ti)
        n_steps = (tf - ti) / dt
        for i in np.arange(n_steps):
            self.update(dt)
            self.storage.append([self.t, self.w, self.z,
                                 self.x, self.attitude, self.lamb_z,
                                 self.beta_z, self.s])

        self.storage.sort(key=lambda x: x[0])

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

    def __init__(self, ccd, delta_z, delta_y):
        self.delta_z = delta_z
        self.delta_y = delta_y
        self.ccd = ccd

        #create storage
        self.obs_times = []
        self.telescope_positions = []
        self.times_deep_scan = []

    def reset_memory(self):
        """
        :return: empty all attribute lists from scanner before beginning new scanning period.
        """
        self.obs_times.clear()
        self.telescope_positions.clear()
        self.times_deep_scan.clear()

    def intercept(self, att, star, line = False):
        """
        :param star: source object.
        :param attitude: contains storage list.
        :return: populates scanner.times_to_scan list, before deep_scan is executed.
        """
        for obj in att.storage:
            t = obj[0]
            x_telescope1 = obj[3]
            attitude = obj[4]

            x_srs_telescope1 = ft.xyz(attitude, x_telescope1)
            star_coor_srs = ft.xyz(attitude, star.coor)

            xy_proy_star_srs = np.array([star_coor_srs[0], star_coor_srs[1], 0])
            xz_proy_star_srs = np.array([star_coor_srs[0], 0, star_coor_srs[2]])

            width_angle = 2 * np.arctan2(self.ccd/2, x_srs_telescope1[0])
            height_angle = 2 * np.arctan2(self.delta_z/2, x_srs_telescope1[0])

            width_star = np.arctan2(xy_proy_star_srs[1], xy_proy_star_srs[0])
            height_star = np.arctan2(xz_proy_star_srs[2], xz_proy_star_srs[0])

            if np.abs(width_star) < width_angle:
                if np.abs(height_star) < height_angle:
                    if line is True:
                        if np.abs(star_coor_srs[1] - x_srs_telescope1[1]) < self.delta_y:
                            self.obs_times.append(t)
                            self.telescope_positions.append(ft.lmn(attitude, x_srs_telescope1))
                    else:
                        self.times_deep_scan.append(t)
                        self.times_deep_scan.sort()

        print('times intercepted:', len(self.times_deep_scan))

    def deep_scan(self, att, star, deep_dt=0.001):

        """
        Increases precision of satellite at points where source is intercept by scanner in the CCD.
        :param: attitude: attitude class.
        :param satellite: satellite object.
        :param deep_dt: new step dt fur higher numerical method precision.
        """
        print ('doing deep_scan')

        for t in self.times_deep_scan:
            att.short_reset_to_time(t)
            att.create_storage(t - 0.2, t + 0.2, deep_dt)

        self.intercept(att, star, line = True)

        print('deep_scan done')


def star_finder(att, sky, scanner):
    """
    Finds times at which source transit CCD line of scanner and estimates their position in the BCRS frame.
    :param sky: object
    :param att:object
    :param scanner: object
    :return: scanner.star_positions, scanner.obs_times.
    """
    for star in sky.elements:
        scanner.reset_memory()
        scanner.intercept(att, star)
        if len(scanner.times_deep_scan) != 0:
            scanner.deep_scan(att, star)

def run(days = 1825, dt = 0.2, stars=1):
    """
    :param days: number of days to run
    :param dt: time step
    :return: sky, scan, att
    """
    start_time = time.time()

    sky = Sky(stars)
    scan = Scanner(0.15,0.15,0.01)
    att = Attitude()
    att.create_storage(0, days, dt)
    star_finder(att, sky, scan)

    seconds = time.time() - start_time
    
    print('seconds:', seconds)
    print('star measurements:', len(scan.telescope_positions))
    
    return att, sky, scan
