# -*- coding: utf-8 -*-
"""
Source class implementation in Python


:Authors: mdelvallevaro
          LucaZampier (modifications)
"""

# # Imports
# Global imports
import numpy as np
# Local imports
#from source import Source
import constants as const
import frame_transformations as ft
from satellite import Satellite
import quaternion

class Source:
    """
    | Source class implemented to represent a source object in the sky
    | Examples:

        >>> Bernard = Source("bernard", 269.45, 4.6933 , 547.4506, -802.803, 10362.54, -0.55, 5.9)

    """

    def __init__(self, name, alpha0, delta0, parallax, mu_alpha, mu_delta, g_alpha, g_delta, mu_radial=0):

        """
        The initial parameters of the source are given with this units of mesurement:
        :param alpha0: [deg] -Right ascension
        :param delta0: [deg] -Declination
        :param parallax: [mas] -Parallax
        :param mu_alpha: [mas/yr] -Proper motion in right ascension
        :param mu_delta: [mas/yr] -Proper motion in declination
        :param g_alpha: [mas/yr^2] -Acceleration of proper motion in right ascension
        :param g_delta: [mas/yr^2] -Acceleration of proper motion in declination

        For the model implementation is easier to work with the following transformations:
            * [alpha] = rad
            * [delta] = rad
            * [parallax] = rad
            * [mu_alpha_dx] = rad/days
            * [mu_delta] = rad/days

        """
        self.name = name
        self.__alpha0 = np.radians(alpha0)
        self.__delta0 = np.radians(delta0)
        self.alpha = self.__alpha0
        self.delta = self.__delta0
        self.parallax = parallax * const.rad_per_mas
        self.mu_alpha_dx = mu_alpha * const.rad_per_mas / const.days_per_year * np.cos(self.__delta0)
        self.mu_delta = mu_delta * const.rad_per_mas / const.days_per_year     # from mas/yr to rad/day
        self.mu_radial = mu_radial * self.parallax * const.Au_per_km * const.sec_per_day
        self.g_alpha = g_alpha * const.rad_per_mas / (const.days_per_year)**2
        self.g_delta = g_delta * const.rad_per_mas / (const.days_per_year)**2

    def get_parameters(self, t=0):
        """
        :returns: astrometric parameters at time t (t=0 by default)
        """
        self.set_time(t)
        return np.array([self.alpha, self.delta, self.parallax, self.mu_alpha_dx, self.mu_delta, self.g_alpha, self.g_delta, self.mu_radial])

    def reset(self):
        """
        Reset star position to t=0
        """
        self.alpha = self.__alpha0
        self.delta = self.__delta0

    def set_time(self, t):
        """
        Sets star at position wrt bcrs at time t.

        :param t: [float][days] time
        """
        if t < 0:
            raise ValueError("t [time] sholdn't be negative")

        mu_alpha_dx = self.mu_alpha_dx
        mu_delta = self.mu_delta

        self.alpha = self.__alpha0 + mu_alpha_dx*t
        self.delta = self.__delta0 + mu_delta*t

    def barycentric_direction(self, t):
        """
        Direction unit vector to star from bcrs.

        :param t: [float][days]
        :return: ndarray 3D vector of [floats]
        """
        self.set_time(0)
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

    def compute_u(self, sat, t):
        """
        Compute the topocentric_function direction

        :param satellite: satellite [class object]
        :return: [array] (x,y,z) direction-vector of the star from the satellite's lmn frame.
        """
        # self.set_time(0)  # (float(t))
        param = np.array([self.alpha, self.delta, self.parallax, self.mu_alpha_dx, self.mu_delta, self.g_alpha, self.g_delta])
        p, q, r = ft.compute_pqr(self.alpha, self.delta)
        t_B = t  # + r.transpose() @ b_G / const.c  # # TODO: replace t_B with its real value
        tau=t_B-const.t_ep
        b_G = sat.ephemeris_bcrs(t)  # [Au]
        topocentric = r + tau**2/2 * (p * self.g_alpha + q * self.g_delta) #- b_G * const.Au_per_Au * self.parallax
        norm_topocentric = np.linalg.norm(topocentric)

        return topocentric / norm_topocentric


    def compute_du_ds(self, satellite,p,q,r,q_l,t_l):
        """
        params p,q,r : the vectors defining the frame associated to a source position at reference epoch
        params q_l,t_l : the attitude at time t_l
        returns : du_ds_SRS
        """
        # Equation 73
        r.shape = (3, 1)  # reshapes r
        b_G = satellite.ephemeris_bcrs(t_l)
        tau = t_l - const.t_ep  # + np.dot(r, b_G) / const.c
        # Compute derivatives
        du_ds_CoMRS = [p, q, self.compute_du_dparallax(r, b_G), p*tau, q*tau, p*(tau**2)/2, p*(tau**2)/2]
        # Equation 72
        # should be changed to a pythonic map
        du_ds_SRS = []
        for derivative in du_ds_CoMRS:
            du_ds_SRS.append(ft.lmn_to_xyz(q_l, derivative))
        return np.array(du_ds_SRS)

    def compute_du_dparallax(self,r, b_G):
        """
        | Ref. Paper [LUDW2011]_ eq. [73]
        | Computes :math:`\\frac{du}{d\omega}`

        :param r: barycentric coordinate direction of the source at time t.
         Equivalently it is the third column vector of the "normal triad" of the
         source with respect to the ICRS.
        :param b_G: Spatial coordinates in the BCRS.
        :returns: [array] the derivative du_dw
        """
        if not isinstance(b_G, np.ndarray):
            raise TypeError('b_G has to be a numpy array, instead is {}'.format(type(b_G)))
        if r.shape != (3, 1):
            raise ValueError('r.shape should be (1, 3), instead it is {}'.format(r.shape))
        if len(b_G.flatten()) != 3:
            raise ValueError('b_G should have 3 elements, instead has {}'.format(len(b_G.flatten())))
        if len((r @ r.T).flatten()) != 9:
            raise Error("rr' should have 9 elements! instead has {} elements".format(len((r @ r.T).flatten())))
        b_G.shape = (3, 1)
        # r.shape = (1, 3)
        du_dw = -(np.eye(3) - r @ r.T) @ b_G / const.Au
        du_dw.shape = (3)  # This way it returns an error if it has to copy data
        return du_dw  # np.ones(3)  #

    def topocentric_angles(self, satellite, t):
        """
        Calculates the angles of movement of the star from bcrs.

        :param satellite: satellite object
        :param t: [days]
        :return: alpha, delta, delta alpha, delta delta [mas]
        """

        u_lmn_unit = self.unit_topocentric_function(satellite, t)
        alpha_obs, delta_obs = ft.vector_to_alpha_delta(u_lmn_unit)

        if alpha_obs < 0:
            alpha_obs = (alpha_obs + 2*np.pi)

        delta_alpha_dx_mas = (alpha_obs - self.__alpha0) * np.cos(self.__delta0) / const.rad_per_mas
        delta_delta_mas = (delta_obs - self.__delta0) / const.rad_per_mas

        return alpha_obs, delta_obs, delta_alpha_dx_mas, delta_delta_mas  # mas

class Calc_source:
    """
    Contains the calculated parameters per source
    """
    def __init__(self,name=None, source=None , obs_times=[]):
        """
        Data structure containing our computed parameters for the source in
        question.

        :param name: [string] the name of the source
        :param obs_times: [list or array] of the observed times for this source
        :param source_params: [list or array] alpha, delta, parallax, mu_alpha, mu_delta
        :param mu_radial: [float] radial velocity of the source (appart since we
         do not solve for radial velocity)
        :param source: [source] instead of most of the above parameters we can
         provide a source object instead and take the data from it.
         Manually providing the parameters will override the source parameter

        see :class:`source.Source`

        >>> calc_source = Calc_source('calc_sirio', [1, 2.45, 12], [1, 2, 3, 4, 5], 6)
        >>> calc_source = Calc_source(obs_times=[1, 2, 3], source=sirio)  # where sirio is a source object

        """

        if source is not None:
            name = 'Calc_' + source.name
            params = source.get_parameters()
            source_params = params[0:]
        self.name = name
        self.source=source
        self.obs_times = obs_times  # times at which it has been observed
        self.s_old = self.source.get_parameters()
        self.errors = []


    def set_params(self, params):
        self.s_old = self.source.get_parameters()
        self.source.set_params(params)


    def compute_u(self,sat,t):
        return self.source.compute_u(sat,t)

    def set_params(self, params):
        self.s_params = params
        self.s_old = [self.s_params]


def calculated_field_angles(calc_source, attitude, sat, t, double_telescope=False):
    """
    | Ref. Paper [LUDW2011]_ eq. [12]-[13]
    | Return field angles according to Lindegren eq. 12. See :meth:`compute_field_angles`

    :param source: [Calc_source]
    :param attitude: [quaternion] attitude at time t
    :param sat: [Satellite]
    :param t: [float] time at which we want the angles
    :param double_telescope: [bool] If true, uses the model with two telescopes
    :returns:
        * eta: along-scan field angle (== phi if double_telescope = False)
        * zeta: across-scan field angle
    """
    alpha, delta, parallax, mu_alpha, mu_delta, g_alpha, g_delta= calc_source.source.get_parameters()
    params = np.array([alpha, delta, parallax, mu_alpha, mu_delta, g_alpha, g_delta ])

    Cu = calc_source.compute_u(sat, t)  # u in CoMRS frame
    Su = ft.lmn_to_xyz(attitude, Cu)  # u in SRS frame

    eta, zeta = compute_field_angles(Su, double_telescope)
    return eta, zeta
