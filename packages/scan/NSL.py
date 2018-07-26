# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:59:19 2018

@author: vallevaro

http://docs.astropy.org/en/stable/coordinates/index.html#module-astropy.coordinates
"""

import frame_transformations as ft
from quaternion import Quaternion
from numba import jit
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
            mualpha = np.random.uniform(0, 0.1)
            mudelta = np.random.uniform(0, 0.1)
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

    def __init__(self, alpha, delta, mualpha=0, mudelta=0, parallax=1):
        # make alpha delta attributes
        # time dependence, get direction at time __t. Create a function coor that does all of this and
        # is able to implement relativistic effects.
        self.coor = ft.xyz(alpha, delta)
        self.velocity = np.array([mualpha, mudelta, parallax])

class Satellite:

    def __init__(self, *args):
        self.init_parameters(*args)

    def init_parameters(self, S=4.036, epsilon=np.radians(23.26), xi=np.radians(55), wz=120):
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

    @jit
    def update(self, dt):
        """
        :param dt: time step to calculate derivatibles of functions
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

        w_magnitude = np.linalg.norm(self.w)
        d_zheta = w_magnitude * dt
        delta_quat = ft.rotation_to_quat(self.w, d_zheta/2.) # w is not in bcrs frame.
        self.attitude = delta_quat * self.attitude

        # x axis
        x_quat = Quaternion(0, self.x[0], self.x[1], self.x[2])
        x_quat = delta_quat* x_quat* delta_quat.conjugate()
        self.x = x_quat.to_vector()

    @jit
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

    def __init__(self, ccd=0.3, delta_z=0.2, delta_y=0.01):
        self.delta_z = delta_z
        self.delta_y = delta_y
        self.ccd = ccd

        #create storage
        self.obs_times = []
        self.stars_positions = []
        self.times_deep_scan = []

    def reset_memory(self):
        """
        :return: empty all attribute lists from scanner before beginning new scanning period.
        """
        self.obs_times.clear()
        self.stars_positions.clear()
        self.times_deep_scan.clear()

    @jit
    def intercept(self, att, star):
        """
        :param star: source object.
        :param attitude: contains storage list.
        :return: populates scanner.times_to_scan list, before deep_scan is executed.
        """
        print ('intercepting')
        for idx, obj in enumerate(att.storage):
            t = obj[0]
            x_telescope1 = obj[3]
            attitude = obj[4]

            x_srs_telescope1 = ft.srs(attitude, x_telescope1)
            star_coor_srs = ft.srs(attitude, star.coor)

            xy_proy_star_srs = np.array([star_coor_srs[0], star_coor_srs[1], 0])
            xz_proy_star_srs = np.array([star_coor_srs[0], 0, star_coor_srs[2]])

            width_angle = 2 * np.arctan2(self.ccd/2, x_srs_telescope1[0])
            height_angle = 2 * np.arctan2(self.delta_z/2, x_srs_telescope1[0])

            width_star = np.arctan2(xy_proy_star_srs[1], xy_proy_star_srs[0])
            height_star = np.arctan2(xz_proy_star_srs[2], xz_proy_star_srs[0])

            if np.abs(width_star) < width_angle:
                if np.abs(height_star) < height_angle:
                    self.times_deep_scan.append([t, idx])
                    self.times_deep_scan.sort()
                    print ('scanner.times_deep_scan filled')
    

    def deep_scan(self, att, deep_dt=0.001):
        """
        Increases precision of satellite at points where source is intercept by scanner in the CCD.
        :param: attitude: attitude class.
        :param satellite: satellite object.
        :param deep_dt: new step dt fur higher numerical method precision.
        """
        print ('doing deep_scan')
        target_times = [i[0] for i in self.times_deep_scan]
        i = 0
        for t in target_times:
            i = i +1
            print (i, len(self.times_deep_scan))
            att.create_storage(t - 0.1, t + 0.1, deep_dt)


def star_finder(att, sky, scanner):
    """
    Finds times at which source transit CCD line of scanner and estimates their position in the BCRS frame.
    :param sky: object
    :param att:object
    :param scanner: object
    :return: scanner.star_positions, scanner.obs_times.
    """
    scanner.reset_memory()
    for star in sky.elements:
        scanner.intercept(att, star)

        if len(scanner.times_deep_scan) != 0:
            scanner.deep_scan(att)
            indexes = [i[1] for i in scanner.times_deep_scan]
            for index in  indexes:
                obj = att.storage[index]
                t = obj[0]
                attitude = obj[4]
                x_telescope1 = obj[3]
                x_srs_telescope1 = ft.srs(attitude, x_telescope1)

                star_srs_coor = ft.srs(attitude, star.coor)
                height_angle = 2* np.arctan2(scanner.delta_z/2, x_srs_telescope1[0])

                xy_star = np.array([star_srs_coor[0], star_srs_coor[1], 0])
                xz_star = np.array([star_srs_coor[0], 0, star_srs_coor[2]])

                width_star = np.arctan2(xy_star[1], xy_star[0])
                height_star = np.arctan2(xz_star[2], xz_star[0])

                if np.abs(width_star) < height_angle:
                    if np.abs(height_star) < scanner.delta_z:
                        if np.abs(star_srs_coor[1] - x_srs_telescope1[1]) < scanner.delta_y:

                            scanner.obs_times.append(t)
                            scanner.stars_positions.append(ft.bcrs(attitude, x_srs_telescope1))
                            print ('found star at:', t, index)

def run(days, dt):
    """

    :param days: number of days to run
    :param dt: time step
    :return: sky, scan, att
    """
    start_time = time.time()

    sky = Sky(1)
    scan = Scanner()
    att = Attitude()
    att.create_storage(0, days, dt)
    star_finder(att, sky, scan)

    seconds = time.time() - start_time
    
    print('seconds:', seconds)
    print('star measurements:', len(scan.stars_positions))
    
    return sky, scan, att
