"""
agis_helpers.py
functions that uses the classes scanner, source, satellite but don't belong to a
given file yet
author: LucaZampieri
"""

# # Imports
# Global imports
import numpy as np
# Local imports
import constants as const
import helpers as helpers
import frame_transformations as ft
from quaternion import Quaternion
from source import Source
from satellite import Satellite
from scanner import Scanner
from source import get_Cu


def rotation_matrix_from_alpha_delta(source, sat, t):
    Cu = source.unit_topocentric_function(sat, t)
    Su = np.array([1, 0, 0])
    r = ft.get_rotation_matrix(Cu, Su)
    return r


def attitude_from_alpha_delta(source, sat, t):
    Cu = source.unit_topocentric_function(sat, t)
    Su = np.array([1, 0, 0])
    vector, angle = ft.get_rotation_vector_and_angle(Cu, Su)
    return ft.rotation_to_quat(vector, angle)


def observed_field_angles(source, sat, t):
    """
    Return field angles according to Lindegren eq. 12
    eta: along-scan field angle
    """
    Gamma_c = 0  # angle between the two scanners # TODO: implement gamma_c
    Cu = source.unit_topocentric_function(sat, t)  # u in CoMRS frame
    alpha, delta, _, _ = source.topocentric_angles(sat, t)
    # Su = ft.lmn_to_xyz(sat.func_attitude(t), Cu)  # u in SRS frame
    r = rotation_matrix_from_alpha_delta(source, sat, t)
    # print(r@Cu.T)
    Su = np.array(r@Cu.T)
    attitude = attitude_from_alpha_delta(source, sat, t)
    # Su = ft.lmn_to_xyz(attitude, Cu)
    # print(Su.shape)
    # print(Su)
    Su_x = Su[0]
    Su_y = Su[1]
    Su_z = Su[2]

    phi = np.arctan2(Su_y, Su_x)
    if phi >= np.pi or phi < -np.pi:
        raise ValueError('phi should be -pi <= phi < pi, instead it is: {}'.format(phi))
    zeta = np.arctan2(Su_z, np.sqrt(Su_x**2+Su_y**2))
    field_index = np.sign(phi)
    eta = phi - field_index * Gamma_c / 2
    return eta, zeta


def compute_field_angles(calc_source, source, sat, t):
    """
    Return field angles according to Lindegren eq. 12
    eta: along-scan field angle
    """
    Gamma_c = 0  # angle between the two scanners # TODO: implement gamma_c
    alpha, delta, parallax, mu_alpha, mu_delta = calc_source.s_params[:]
    params = np.array([alpha, delta, parallax, mu_alpha, mu_delta, calc_source.mu_radial])
    Cu = get_Cu(params, sat, t)  # u in CoMRS frame
    # Su = ft.lmn_to_xyz(sat.func_attitude(t), Cu)  # u in SRS frame
    r = rotation_matrix_from_alpha_delta(source, sat, t)
    Su = np.array(r@Cu.T)

    Su_x = Su[0]
    Su_y = Su[1]
    Su_z = Su[2]
    phi = np.arctan2(Su_y, Su_x)
    if phi >= np.pi or phi < -np.pi:
        raise ValueError('phi should be -pi <= phi < pi, instead it is: {}'.format(phi))
    zeta = np.arctan2(Su_z, np.sqrt(Su_x**2+Su_y**2))
    field_index = np.sign(phi)
    eta = phi - field_index * Gamma_c / 2
    return eta, zeta


"""
def get_Cu(astro_parameters, sat, t):

    Compute the topocentric_function direction i.e. ũ
    The horizontal coordinate system, also known as topocentric coordinate
    system, is a celestial coordinate system that uses the observer's local
    horizon as the fundamental plane. Coordinates of an object in the sky are
    expressed in terms of altitude (or elevation) angle and azimuth.

    :return: [array] (x,y,z) direction-vector of the star from the satellite's lmn frame.

    # if not isinstance(satellite, Satellite):
    #     raise TypeError('Expected Satellite, but got {} instead'.format(type(satellite)))
    alpha, delta, parallax, mu_alpha_dx, mu_delta, mu_radial = astro_parameters[:]
    p, q, r = ft.compute_pqr(alpha, delta)

    mu_alpha_dx = mu_alpha_dx * const.rad_per_mas / const.days_per_year   # mas/yr to rad/day
    mu_delta = mu_delta * const.rad_per_mas / const.days_per_year  # mas/yr to rad/day
    # km/s to aproximation rad/day
    parallax = parallax * const.rad_per_mas
    mu_radial = parallax * mu_radial * const.km_per_pc * const.sec_per_day

    # WARNING: barycentric coordinate is not defined in the same way!
    # topocentric_function direction
    t_B = t  # + r.transpose() @ b_G / const.c  # # TODO: replace t_B with its real value
    b_G = sat.ephemeris_bcrs(t)
    topocentric = r + t * (p * mu_alpha_dx + q * mu_delta + r * mu_radial) - parallax * b_G * const.AU_per_pc
    norm_topocentric = np.linalg.norm(topocentric)

    return topocentric / norm_topocentric
"""


def compute_mnu(eta, xi):
    """
    return column vectors of the S'[m_l, n_l, u_l] matrix
    """
    phi = eta  # # TODO: implement the correct version (phi != eta)
    # S_mnu = np.zeros((3,3))
    m_l = np.array([-np.sin(phi), np.cos(phi), 0])
    n_l = np.array([-np.sin(xi)*np.cos(phi), np.sin(xi)*np.sin(phi), np.cos(xi)])
    u_l = np.array([np.cos(xi)*np.cos(phi), np.cos(xi)*np.sin(phi), np.sin(xi)])
    return np.array([m_l, n_l, u_l])


# Draft of helper functions
def phi(source, sat, t):
    """
    Calculates the diference between the x-axis of the satellite and the direction vector to the star.
    Once this is calculated, it checks how far away is in the alpha direction (i.e. the y-component) wrt IRS.
    :param source: Source [object]
    :param sat: Satellite [object]
    :param t: time [float][days]
    :return: [float] angle, alpha wrt IRS.
    """
    t = float(t)
    u_lmn_unit = source.unit_topocentric_function(sat, t)
    direction_lmn = u_lmn_unit - sat.func_x_axis_lmn(t)
    direction_xyz = ft.lmn_to_xyz(sat.func_attitude(t), direction_lmn)
    phi = np.arcsin(direction_xyz[1])
    eta = np.arcsin(direction_xyz[2])
    # phi = np.arctan2(direction_xyz[1], direction_xyz[0])
    # xi = np.arctan2(direction_xyz[2], np.sqrt(direction_xyz[0] ** 2 + direction_xyz[1] ** 2))
    return phi, eta


################################################################################
# Unused functions

def eta0_fng(mu, f, n, g):
    # = eta0_ng
    # TODO: define Y_FPA, F
    return -Y_FPA[n, g]/F


def xi0_fng(mu, f, n, g):
    """
    :attribute X_FPA[n]: physical AC coordinate of the nominal center of the nth CCD
    :attribute Y_FPA[n,g]: physical AL coordinate of the nominal observation line for gate g on the nth CCD
    :attribute Xcentre_FPA[f]:
    """
    mu_c = 996.5
    p_AC = 30  # [micrometers]
    return -(X_FPA[n] - (mu - mu_c) * p_AC - Xcenter_FPA[f])/F
