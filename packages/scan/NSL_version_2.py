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
from mpl_toolkits.mplot3d import Axes3D
from astropy import constants
from astropy import units


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

#vega = Source("vega", 279.2333, 38.78, 128.91, 201.03, 286.23, -13.9)
#proxima = Source("proxima",217.42, -62, 768.7, 3775.40, 769.33, 21.7)
#sirio = Source("sirio", 101.28, -16.7161, 379.21, -546.05, -1223.14, -7.6)

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


    def reset(self):
        self.alpha = self.__alpha0
        self.delta = self.__delta0

    def set_time(self, t): #implement here relativistic effects
        mu_alpha_dx = self.mu_alpha_dx * 4.8473097e-9 / 365 #from mas/yr to rad/day
        mu_delta = self.mu_delta * 4.848136811095e-9 / 365  #from mas/yr to rad/day
        self.alpha = self.__alpha0 + mu_alpha_dx*t  #rad
        self.delta = self.__delta0 + mu_delta*t     #rad

    def barycentric_direction(self, t):
        """
        alpha: rad
        delta: rad
        parallax: mas
        :param t: days
        :return: vector in parsecs
        """
        self.set_time(t)
        u_bcrs = ft.cartesian_coord(self.alpha, self.delta, self.parallax)
        return u_bcrs   #in pc

    def topocentric_direction(self, satellite, t):

        self.set_time(t)
        p, q, r = ft.pqr(self.alpha, self.delta)
        mastorad = 2*np.pi/(1000*360*3600)
        kmtopc = 3.24078e-14
        sectoday = 3600*24

        mu_alpha_dx = self.mu_alpha_dx*mastorad/365   #mas/yr to rad/day
        mu_delta = self.mu_delta*mastorad/365  #mas/yr to rad/day
        mu_radial = self.parallax*mastorad*self.mu_radial*kmtopc*sectoday #km/s to aproximation rad/day

        u_lmn = self.barycentric_direction(0) + t*(p*mu_alpha_dx+ q*mu_delta + r*mu_radial) - satellite.ephemeris_bcrs(t) #pc
        u_lmn_unit  = u_lmn/np.linalg.norm(u_lmn)

        alpha_obs, delta_obs, radius = ft.cartesian_to_polar(u_lmn_unit) #rad, rad, pc=1

        if alpha_obs < 0:
            alpha_obs = alpha_obs + 2*np.pi

        delta_alpha_dx_mas = (alpha_obs - self.__alpha0)*np.cos(self.__delta0)/mastorad
        delta_delta_mas = (delta_obs - self.__delta0)/mastorad

        return delta_alpha_dx_mas, delta_delta_mas


class Satellite:

    def __init__(self, *args):
        self.init_parameters(*args)
        self.orbital_period = 365
        self.orbital_radius = 1.0

    def init_parameters(self, S=4.036, epsilon=np.radians(23.26), xi=np.radians(45), wz=120):
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
        :return: 3d np.array
        """
        b_x_bcrs = self.orbital_radius*np.cos(2*np.pi/self.orbital_period*t)*np.cos(self.epsilon)
        b_y_bcrs = self.orbital_radius*np.sin(2*np.pi/self.orbital_period*t)*np.cos(self.epsilon)
        b_z_bcrs = self.orbital_radius*np.sin(2*np.pi/self.orbital_period*t)*np.sin(self.epsilon)

        bcrs_ephemeris_satellite = np.array([b_x_bcrs, b_y_bcrs, b_z_bcrs])

        return bcrs_ephemeris_satellite*4.8481705933824e-6 #in parsecs

class Attitude(Satellite):
    """
    Child class to Satellite.
    """
    def __init__(self, ti, tf, dt):
        Satellite.__init__(self)
        self.init_state()
        self.storage = []
        self.__create_storage(ti, tf, dt)
        self.__attitude_spline()

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

    def reset(self, ti, tf, dt):
        """
        :return: reset satellite to initialization status
        """
        self.init_state()
        self.storage.clear()
        self.__create_storage(ti, tf, dt)
        self.__attitude_spline()

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

        self.s_x= interpolate.InterpolatedUnivariateSpline(t_list, x_list)
        self.s_y = interpolate.InterpolatedUnivariateSpline(t_list, y_list)
        self.s_z = interpolate.InterpolatedUnivariateSpline(t_list, z_list)
        self.s_w = interpolate.InterpolatedUnivariateSpline(t_list, w_list)

    def get_attitude(self, t):
        attitude = Quaternion(float(self.s_w(t)), float(self.s_x(t)), float(self.s_y(t)), float(self.s_z(t)))
        return attitude

    def get_xaxis_lmn(self, t):
        return None

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
        list_element = temp_list[-1]
        self.t = list_element[0]
        self.w = list_element[1]
        self.z = list_element[2]
        self.x = list_element[3]
        self.attitude = list_element[4]
        self.s = list_element[5]

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
            self.long_reset_to_time(ti, dt)

        #if len(self.storage) > 0:
         #   raise Warning('storage is not empty')
         #   self.short_reset_to_time(ti)

        n_steps = (tf - ti) / dt
        self.storage.append([self.t, self.w, self.z,
                             self.x, self.attitude, self.s])
        for i in np.arange(n_steps):
            self.update(dt)
            self.storage.append([self.t, self.w, self.z,
                                 self.x, self.attitude, self.s])

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
