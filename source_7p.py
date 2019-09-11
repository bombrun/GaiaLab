# # Imports
# Global imports
import numpy as np
import quaternion

# Local imports
import gaialab.scanner.constants as const
import gaialab.scanner.frame_transformations as ft
from gaialab.scanner.satellite import Satellite


def compute_topocentric_direction(astro_parameters, sat, t):

    alpha, delta, parallax, mu_alpha_dx, mu_delta, mu_radial, g_alpha, g_delta = astro_parameters[:]
    p, q, r = ft.compute_pqr(alpha, delta)
    t_B = t + r.transpose() @ b_G /const.c # + r.transpose() @ b_G / const.c  # # TODO: replace t_B with its real value
    b_G = sat.ephemeris_bcrs(t)  # [Au]
    topocentric = r + t * (p * mu_alpha_dx + q * mu_delta + r * mu_radial) - b_G * const.Au_per_Au * parallax
    norm_topocentric = np.linalg.norm(topocentric)

    return topocentric / norm_topocentric

class Source:

    def __init__(self, name, alpha0, delta0, parallax, mu_alpha, mu_delta, radial_velocity,
                     func_color=(lambda t: 0), mean_color=0):

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

    self.set_time(t)
    return np.array([self.alpha, self.delta, self.parallax, self.mu_alpha_dx, self.mu_delta, self.mu_radial])

def reset(self):

    self.alpha = self.__alpha0
    self.delta = self.__delta0

def set_time(self, t):

    if t < 0:
        raise ValueError("t [time] sholdn't be negative")

    mu_alpha_dx = self.mu_alpha_dx
    mu_delta = self.mu_delta

    self.alpha = self.__alpha0 + mu_alpha_dx*t
    self.delta = self.__delta0 + mu_delta*t

def barycentric_direction(self, t):

    self.set_time(0)
    u_bcrs_direction = ft.polar_to_direction(self.alpha, self.delta)
    return u_bcrs_direction  # no units, just a unit direction

def barycentric_coor(self, t):
    "
    self.set_time(t)
    u_bcrs = ft.adp_to_cartesian(self.alpha, self.delta, self.parallax)
    return u_bcrs

def unit_topocentric_function(self, satellite, t):

    # self.set_time(0)  # (float(t))
    param = np.array([self.__alpha0, self.__delta0, self.parallax, self.mu_alpha_dx, self.mu_delta, self.mu_radial])
    return compute_topocentric_direction(param, satellite, t)

def topocentric_angles(self, satellite, t):
    

    u_lmn_unit = self.unit_topocentric_function(satellite, t)
    alpha_obs, delta_obs = ft.vector_to_alpha_delta(u_lmn_unit)

    if alpha_obs < 0:
        alpha_obs = (alpha_obs + 2*np.pi)

    delta_alpha_dx_mas = (alpha_obs - self.__alpha0) * np.cos(self.__delta0) / const.rad_per_mas
    delta_delta_mas = (delta_obs - self.__delta0) / const.rad_per_mas

    return alpha_obs, delta_obs, delta_alpha_dx_mas, delta_delta_mas  # mas
