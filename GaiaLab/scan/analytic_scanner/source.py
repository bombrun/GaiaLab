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
import constants as const
import frame_transformations as ft
from satellite import Satellite
import quaternion


def compute_topocentric_direction(astro_parameters, sat, t):
    """
    Compute the topocentric_function direction i.e. ũ
    The horizontal coordinate system, also known as topocentric coordinate
    system, is a celestial coordinate system that uses the observer's local
    horizon as the fundamental plane. Coordinates of an object in the sky are
    expressed in terms of altitude (or elevation) angle and azimuth.

    :param astro_parameters: [alpha, delta, parallax, mu_alpha_dx, mu_delta, mu_radial]
     [rads] the astrometric parameters
    :param sat: [Satellite]
    :param t: [float][days] time at which we want the topocentric function
    :return: [array] (x,y,z) direction-vector of the star from the satellite's lmn frame.
     (CoMRS)
    """
    alpha, delta, parallax, mu_alpha_dx, mu_delta, mu_radial = astro_parameters[:]
    p, q, r = ft.compute_pqr(alpha, delta)
    t_B = t  # + r.transpose() @ b_G / const.c  # # TODO: replace t_B with its real value
    b_G = sat.ephemeris_bcrs(t)  # [Au]
    topocentric = r + t * (p * mu_alpha_dx + q * mu_delta + r * mu_radial) - b_G * const.Au_per_Au * parallax
    norm_topocentric = np.linalg.norm(topocentric)

    return topocentric / norm_topocentric


class Source:
    """
    | Source class implemented to represent a source object in the sky
    | Examples:

        >>> vega = Source("vega", 279.2333, 38.78, 128.91, 201.03, 286.23, -13.9)
        >>> proxima = Source("proxima",217.42, -62, 768.7, 3775.40, 769.33, 21.7)
        >>> sirio = Source("sirio", 101.28, -16.7161, 379.21, -546.05, -1223.14, -7.6)
    """

    def __init__(self, name, alpha0, delta0, parallax, mu_alpha, mu_delta, radial_velocity,
                 func_color=(lambda t: 0), mean_color=0):
        """
        :param alpha0: [deg]
        :param delta0: [deg]
        :param parallax: [mas]
        :param mu_alpha: [mas/yr]
        :param mu_delta: [mas/yr]
        :param radial_velocity: [km/s]
        :param func_color: function representing the color of the source in nanometers
        :param mean_color: mean color observed by satellite

        Transforms in rads/day or rads, i.e. we got:
            * [alpha] = rads
            * [delta] = rads
            * [parallax] = rads
            * [mu_alpha_dx] = rads/days
            * [mu_delta] = rads/days
            * [mu_radial] = rads/days
        """
        self.name = name
        self.__alpha0 = np.radians(alpha0)
        self.__delta0 = np.radians(delta0)
        self.parallax = parallax * const.rad_per_mas
        self.mu_alpha_dx = mu_alpha * const.rad_per_mas / const.days_per_year * np.cos(self.__delta0)
        self.mu_delta = mu_delta * const.rad_per_mas / const.days_per_year     # from mas/yr to rad/day
        self.mu_radial = radial_velocity * self.parallax * const.Au_per_km * const.sec_per_day
        self.alpha = self.__alpha0
        self.delta = self.__delta0

        # For the source color
        self.func_color = func_color
        self.mean_color = mean_color

    def get_parameters(self, t=0):
        """
        :returns: astrometric parameters at time t (t=0 by default)
        """
        self.set_time(t)
        return np.array([self.alpha, self.delta, self.parallax, self.mu_alpha_dx, self.mu_delta, self.mu_radial])

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

    def unit_topocentric_function(self, satellite, t):
        """
        Compute the topocentric_function direction

        :param satellite: satellite [class object]
        :return: [array] (x,y,z) direction-vector of the star from the satellite's lmn frame.
        """
        # self.set_time(0)  # (float(t))
        param = np.array([self.__alpha0, self.__delta0, self.parallax, self.mu_alpha_dx, self.mu_delta, self.mu_radial])
        return compute_topocentric_direction(param, satellite, t)

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
