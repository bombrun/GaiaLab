"""
Satellite class implementation in Python


@author: mdelvallevaro
"""

# # Imports
# Global imports
import numpy as np
from scipy import interpolate
from scipy.interpolate import splrep
# Local imports
import constants as const
import frame_transformations as ft
from quaternion import Quaternion


class Satellite:
    """
    Class Object, that represents a satellite, e.g. Gaia.
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

    :param S: change in the z-axis of satellite wrt solar longitudinal angle.
     [float]
    :param epsilon: ecliptical angle [rad]
    :param xi: revolving angle [rad]
    :param wz: z component of inertial spin vector [arcsec/s]
    :action: Sets satellite to initialization status.
    """
    def __init__(self, ti=0, tf=5*const.days_per_year, dt=1/24, k=4, *args):
        """
        :orbital_period: [days]
        :orbital_radius: [AU]
        """
        self.init_parameters(*args)
        self.orbital_period = const.days_per_year
        self.orbital_radius = 1.0
        # Splines for each coordinates i, i_list at each time in t_list of degree k (order = k+1)
        self.spline_order = k

        self.storage = []
        self.__init_state()
        self.__create_storage(ti, tf, dt)
        self.__init_state()
        self.__attitude_spline()

    def init_parameters(self, S=const.S, epsilon=np.radians(const.epsilon), xi=np.radians(const.xi), wz=const.w_z):
        self.S = S

        # obliquity of equator. This is a constant chosen to be 23º 26' 21.448''
        self.epsilon = epsilon

        # "ksi" revolving angle. At any time the z axis is at this constant angle from s⃗ s→.
        # For Gaia, the current choice is 55º.
        self.xi = xi
        self.wz = wz * const.sec_per_day * const.AU_per_pc  # to [rad/day]

        # Nominal longitud of the sun in the ecliptic plane
        self.lambda_dot = 2 * np.pi / const.days_per_year  # [rad/day] (lambda dot set as const)

    def ephemeris_bcrs(self, t):
        """
        Defines the orbit of the satellite around the sun
        Returns the barycentric ephemeris of the Gaia satellite at time t.
        Equivalently written b_G(t)

        :param t: float [days]
        :return: 3D np.array [AU]!
        """
        # Assuming it is a circle tilted by epsilon:
        b_x_bcrs = self.orbital_radius*np.cos(2*np.pi/self.orbital_period*t)*np.cos(self.epsilon)
        b_y_bcrs = self.orbital_radius*np.sin(2*np.pi/self.orbital_period*t)*np.cos(self.epsilon)
        b_z_bcrs = self.orbital_radius*np.sin(2*np.pi/self.orbital_period*t)*np.sin(self.epsilon)

        bcrs_ephemeris_satellite = np.array([b_x_bcrs, b_y_bcrs, b_z_bcrs])
        return bcrs_ephemeris_satellite

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
        delta_quat = Quaternion(vector=self.w, angle=d_zheta/2.)  # w is not in bcrs frame.
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
            x_list.append(obj[4].x)
            y_list.append(obj[4].y)
            z_list.append(obj[4].z)
            w_list.append(obj[4].w)

        # Splines for each coordinates i, i_list at each time in t_list of degree k (order = k+1)
        self.s_w = interpolate.InterpolatedUnivariateSpline(t_list, w_list, k=self.spline_order)
        self.s_x = interpolate.InterpolatedUnivariateSpline(t_list, x_list, k=self.spline_order)
        self.s_y = interpolate.InterpolatedUnivariateSpline(t_list, y_list, k=self.spline_order)
        self.s_z = interpolate.InterpolatedUnivariateSpline(t_list, z_list, k=self.spline_order)

        # Alternativ efor the splines
        # self.s_w_tck = splrep(t_list, w_list, s=0, k=self.spline_order)
        # self.s_x_tck = splrep(t_list, x_list, s=0, k=self.spline_order)
        # self.s_y_tck = splrep(t_list, y_list, s=0, k=self.spline_order)
        # self.s_z_tck = splrep(t_list, z_list, s=0, k=self.spline_order)

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
            stored in: satellite.storage
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
