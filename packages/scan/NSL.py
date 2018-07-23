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
        # time dependence, get direction at time t. Create a function coor that does all of this and
        # is able to implement relativistic effects.
        self.coor = ft.xyz(alpha, delta)
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

    def init_parameters(self, S=4.036, epsilon=np.radians(23.26), xi=np.radians(45), wz=120):
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
        self.S = S
        self.epsilon = epsilon
        self.xi = xi
        self.wz = wz * (60 * 60 * 24. / 206264.8062470946)  # to [rad/day]

class Attitude(Satellite):

    def __init__(self, ti, tf, dt):
        Satellite.__init__(self)
        self.empty()
        self.t = 0
        self.lambda_ = 0
        self.beta_ = 0
        self.nu = 0
        self.omega = 0
        self.z_ = np.array([0., 0., 1.])
        self.x_ = np.array([1., 0., 0.])
        self.w_ = np.array([0., 0., 1.])
        self.l_, self.j_, self.k_ = ft.ljk(self.epsilon)
        self.s_ = self.l_ * np.cos(self.lambda_) + self.j_ * np.sin(self.lambda_)
        self.lambda_dot = 2 * np.pi / 365

        q1 = Quaternion(np.cos(self.epsilon / 2), np.sin(self.epsilon / 2), 0, 0)
        q2 = Quaternion(np.cos(self.lambda_ / 2), 0, 0, np.sin(self.lambda_ / 2))
        q3 = Quaternion(np.cos((self.nu - np.pi / 2.) / 2), np.sin((self.nu - np.pi / 2.) / 2), 0, 0)
        q4 = Quaternion(np.cos((np.pi / 2. - self.xi) / 2), 0, np.sin((np.pi / 2. - self.xi) / 2), 0)
        q5 = Quaternion(np.cos(self.omega / 2.), 0, 0, np.sin(self.omega / 2.))

        q_total = q1 * q2 * q3 * q4 * q5
        self.attitude = q_total * Quaternion(0, 0, 0, 1) * q_total.conjugate()

        self.create_storage(ti, tf, dt)

    def empty(self):
        self.storage = []

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
        dL = self.lambda_dot * dt
        self.lambda_ = self.lambda_ + dL

        # Updates Nu
        nu_dot = self.lambda_dot * (np.sqrt(self.S ** 2 - np.cos(self.nu) ** 2) + np.cos(self.xi) * np.sin(self.nu)) / np.sin(self.xi)
        d_nu = nu_dot * dt
        self.nu = self.nu + d_nu

        # LatitudeAngles
        lamb_z = self.lambda_ + np.arctan(np.tan(self.xi) * np.cos(self.nu))
        beta_z = np.arcsin(np.sin(self.xi) * np.sin(self.nu))

        # Updates Omega
        omega_dot = self.wz - nu_dot * np.cos(self.xi) - self.lambda_ * np.sin(self.xi) * np.sin(self.nu)
        d_omega = omega_dot * dt
        self.omega = self.omega + d_omega

        # Calculates coordinates and then calculates s-vector
        self.s_ = self.l_ * np.cos(self.lambda_) + self.j_ * np.sin(self.lambda_)

        # Calculates z-axis from cross product: delta_z = (k x z)*lambda_dot + (s x z)dNu
        z_dot_ = np.cross(self.k_, self.z_) * self.lambda_dot + np.cross(self.s_, self.z_) * nu_dot
        dz_ = z_dot_ * dt
        self.z_ = self.z_ + dz_
        self.z_ = self.z_ / np.linalg.linalg.norm(self.z_)

        # Updates inertial rotation vector
        self.w_ = self.k_ * self.lambda_dot + self.s_ * nu_dot + self.z_ * omega_dot

        # Calculates new attitude by delta_quat
        w_magnitude = np.sqrt((self.w_[0]) ** 2 + (self.w_[0]) ** 2 + (self.w_[0]) ** 2)
        d_zheta = w_magnitude * dt
        delta_quat = ft.rotation_to_quat(self.w_, d_zheta)
        self.attitude = delta_quat * self.attitude

        # x axis
        x_quat = self.attitude * Quaternion(0, 1, 0, 0) * self.attitude.conjugate()
        self.x_ = x_quat.to_vector()

    def create_storage(self, ti, tf, dt):
        '''
        Creates data necessary for step numerical methods performed in builtin method .update()
        Args:
            ti (float): integrating time lower limit [days]
            tf (float): integrating time upper limit [days]
            dt (float): step discretness of integration.
        Notes:
            The data is stored in satellite.storing_list
        '''

        self.t = ti
        n_steps = (tf - ti) / dt
        for i in np.arange(n_steps):
            self.update(dt)
            self.storage.append([self.t, self.w_, self.z_, self.x_, self.attitude])

        self.storage.sort(key=lambda x: x[0])

    def reset_to_time(self, t):
        '''
        Resets satellite to time t, along with all the parameters corresponding to that time.
        Args:
            t (float): time from J2000 [days]
        '''

        temp_list = [obj for obj in self.storage if obj[0] <= t]
        list_element = temp_list[-1]
        self.t = list_element[0]
        self.w_ = list_element[1]
        self.z_ = list_element[2]
        self.x_ = list_element[3]
        self.attitude = list_element[4]

        # Account for decimal loss difference from discreteness from NM.
        delta_t = t % self.t
        self.update(delta_t)

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

        self.times_deep_scan = []

        self.obs_times = []
        self.stars_positions = []

    def empty(self):
        """
        :return: empty all attribute lists from scanner before beginning new scanning period.
        """
        self.obs_times = []
        self.stars_positions = []
        self.times_deep_scan = []

    def intercept(self, att, star):
        """
        :param star: source object.
        :param attitude: contains storage list.
        :return: populates scanner.times_to_scan list, before deep_scan is executed.
        """
        self.empty()         # careful for when there are more than one star.
        for obj in att.storage:
            t = obj[0]
            attitude = obj[4]
            x_telescope1 = obj[3]

            x_srs_telescope1 = ft.srs(attitude, x_telescope1)

            star_coor_srs = ft.srs(attitude, star.coor)
            xy_proy_star_srs = np.array([star_coor_srs[0], star_coor_srs[1], 0])
            xz_proy_star_srs = np.array([star_coor_srs[0], 0, star_coor_srs[2]])

            width_angle = 2 * np.arctan2(self.ccd / 2, x_srs_telescope1[0])
            aperture_angle = 2 * np.arctan2(self.delta_z / 2, x_srs_telescope1[0])

            if np.arccos(np.dot(xy_proy_star_srs, x_srs_telescope1)) < width_angle:
                if np.arccos(np.dot(xz_proy_star_srs, x_srs_telescope1)) < aperture_angle:
                    self.times_deep_scan.append(t)
                    self.times_deep_scan.sort()

    def deep_scan(self, att, deep_dt=0.001):
        """
        Increases precision of satellite at points where source is intercept by scanner in the CCD.
        :param: attitude: attitude class.
        :param satellite: satellite object.
        :param deep_dt: new step dt fur higher numerical method precision.
        """
        for t in self.times_deep_scan:
            att.create_storage(t - 0.5, t + 0.5, deep_dt)


def star_finder(scanner, att, sky):
    """
    Finds times at which source transit CCD line of scanner and estimates their position in the BCRS frame.

    :param scanner: scan objects.
    :param att: attitude object.
    :param sky: sky to be scanned.
    :return:
    """
    for star in sky.elements:
        scanner.intercept(att, star)
        scanner.deep_scan(att)

        for obj in att.storage:
            t = obj[0]
            attitude = obj[4]
            x_telescope1 = obj[3]

            # change frame to SRS for scanning.
            x_srs_telescope1 = ft.srs(attitude, x_telescope1)
            star_srs_coor = ft.srs(attitude, star.coor)

            # scanner parameters
            aperture_angle = np.arctan2(scanner.delta_z / 2, x_srs_telescope1[0])

            # condition for x axis:
            if np.arccos(np.dot(star_srs_coor, x_srs_telescope1)) < aperture_angle:
                # condition for z axis
                if np.abs(star_srs_coor[2] - x_srs_telescope1[2]) < scanner.delta_z:
                    # condition for y axis:
                    if np.abs(star_srs_coor[1] - x_srs_telescope1[1]) < scanner.delta_y:
                        scanner.obs_times.append(t)
                        scanner.stars_positions.append(ft.bcrs(attitude, x_srs_telescope1))


def run():
    start_time = time.time()

    sky = Sky(1)
    scan = Scanner()
    att = Attitude(0, 365, 0.01)
    star_finder(scan, att, sky)

    seconds = time.time() - start_time
    print(seconds)
    print(len(scan.stars_positions))
    return sky, scan, att
