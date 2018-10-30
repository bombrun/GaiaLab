"""
agis_helpers.py
functions that uses the classes scanner, source, satellite but don't belong to a
given file yet
author: LucaZampieri

todo:
    - Rotate the attitude
    - attitude i.e. generate observations (with scanner object but without scanning)
    - with scanner
    - two telescope
    - scaling
    -
    * if we add acceleration what happens
    * add noise to observation
    * QSO
    * signal
"""

# # Imports
# Global imports
import numpy as np
from scipy.interpolate import BSpline
from scipy.interpolate import splev
import matplotlib.pyplot as plt
# Local imports
import constants as const
import helpers as helpers
import frame_transformations as ft
from quaternion import Quaternion
from source import Source
from satellite import Satellite
from scanner import Scanner
from source import get_Cu


def generate_observation_wrt_attitude(attitude):
    """
    returns right ascention and declination corresponding to the direction in
    which the x-vector rotated to *attitude* is pointing
    returns alpha, delta in radians
    """
    artificial_u = ft.rotate_by_quaternion(attitude, [1, 0, 0])
    alpha, delta, radius = ft.vector_to_polar(artificial_u)
    return alpha, delta


def get_basis_Bsplines(knots, coeffs, k, obs_times):
    """
    :returns: arrays of size (#coeffs, #obs_times)
    """
    basis_Bsplines = []
    for j, coeff in enumerate(coeffs):
        bool_array = np.arange(len(coeffs)) == j
        tck_mod = (knots, bool_array, k)
        basis_Bspline = splev(obs_times, tck_mod)
        basis_Bsplines.append(basis_Bspline)
    return np.array(basis_Bsplines)


def extract_coeffs_knots_from_splines(attitude_splines, k):
    """
    :param attitude_splines: list or array of splines of scipy.interpolate.InterpolatedUnivariateSpline
    :returns:
        [array] coeff
        [array] knots
        [array] splines
    """
    att_coeffs, att_knots, att_splines = ([], [], [])
    for i, spline in enumerate(attitude_splines):
        coeffs = spline.get_coeffs()
        internal_knots = spline.get_knots()
        knots = extend_knots(internal_knots, k)  # extend the knots to have all the needed ones
        att_coeffs.append(coeffs)
        att_knots.append(knots)
        att_splines.append(BSpline(knots, coeffs, k))
    return np.array(att_coeffs), np.array(att_knots), np.array(internal_knots), np.array(att_splines)


def extract_coeffs_knots_from_splines2(splines_tck, k):
    """
    :param attitude_splines: list or array of splines of scipy.interpolate.InterpolatedUnivariateSpline
    :returns:
        [array] coeff
        [array] knots
        [array] splines
    """
    att_coeffs, att_knots, att_splines, ks = ([], [], [], [])
    for tck in splines_tck:
        t, c, k = tck
        att_knots.append(t)
        att_coeffs.append(c)
        att_splines.append(BSpline(t, c, k))
        ks.append(k)
    return np.array(att_knots), np.array(att_coeffs), np.array(ks), np.array(att_splines)


def get_times_in_knot_interval(time_array, knots, index, M):
    """
    :param time_array: [numpy array]
    return times in knot interval defined by [index, index+M]
    """
    return time_array[(knots[m] <= time_array) & (time_array <= knots[m+M])]


def get_left_index(knots, t, M):
    """
    :param M: spline order (k+1)
    return the left index corresponding to t i.e. i s.t. t_i < t < t_{i+1}
    """
    left_index_array = np.where(knots < t)
    if not list(left_index_array[0]):
        left_index = 0
    else:
        left_index = left_index_array[0][-1]
        if left_index - M >= 0:
            left_index -= M
    return left_index


def extend_knots(internal_knots, k):
    extended_knots = []
    for i in range(k):
        extended_knots.append(internal_knots[0])
    extended_knots += list(internal_knots)
    for i in range(k):
        extended_knots.append(internal_knots[-1])
    return extended_knots


def rotation_matrix_from_alpha_delta(source, sat, t):
    Cu = source.unit_topocentric_function(sat, t)
    Su = np.array([1, 0, 0])
    r = helpers.get_rotation_matrix(Cu, Su)
    return r


def attitude_from_alpha_delta(source, sat, t):
    Cu = source.unit_topocentric_function(sat, t)
    Su = np.array([1, 0, 0])
    vector, angle = helpers.get_rotation_vector_and_angle(Cu, Su)
    return Quaternion(vector=vector, angle=angle)


def spin_axis_from_alpha_delta(source, sat, t):
    Cu = source.unit_topocentric_function(sat, t)
    Su = np.array([1, 0, 0])
    vector, angle = helpers.get_rotation_vector_and_angle(Cu, Su)
    # vector = vector/np.linalg.norm(vector)
    # satellite_position = sat.ephemeris_bcrs(t)
    return vector


def scanning_y_coordinate(source, sat, t):
    att = get_fake_attitude(source, sat, t)
    y_vec = ft.rotate_by_quaternion(att, [0, 1, 0])
    # vector = vector/np.linalg.norm(vector)
    # satellite_position = sat.ephemeris_bcrs(t)
    return y_vec


def get_fake_attitude(source, sat, t):
    quat1 = attitude_from_alpha_delta(source, sat, t)
    quat2 = Quaternion(vector=np.array([1, 0, 0]), angle=const.sat_angle)
    attitude = quat1 * quat2
    return sat.func_attitude(t)


def observed_field_angles(source, sat, t):
    """
    Return field angles according to Lindegren eq. 12
    eta: along-scan field angle
    zeta: across-scan field angle
    """
    alpha, delta, _, _ = source.topocentric_angles(sat, t)
    Cu = source.unit_topocentric_function(sat, t)  # u in CoMRS frame
    # Su = ft.lmn_to_xyz(sat.func_attitude(t), Cu)  # u in SRS frame
    attitude = attitude_from_alpha_delta(source, sat, t)
    Su = ft.rotate_by_quaternion(attitude, Cu)
    quat2 = Quaternion(vector=Su, angle=const.sat_angle)
    Su = ft.rotate_by_quaternion(quat2, Su)

    eta, zeta = compute_field_angles(Su)
    return eta, zeta


def calculated_field_angles(calc_source, attitude, sat, t):
    """
    Return field angles according to Lindegren eq. 12
    eta: along-scan field angle
    """
    alpha, delta, parallax, mu_alpha, mu_delta = calc_source.s_params[:]
    alpha = alpha  # + mu_alpha*t
    delta = delta  # + mu_delta*t
    params = np.array([alpha, delta, parallax, mu_alpha, mu_delta, calc_source.mu_radial])
    Cu = get_Cu(params, sat, t)  # u in CoMRS frame

    Su = ft.rotate_by_quaternion(attitude, Cu)
    quat2 = Quaternion(vector=Su, angle=const.sat_angle)
    Su = ft.rotate_by_quaternion(quat2, Su)

    eta, zeta = compute_field_angles(Su)
    return eta, zeta


def compute_field_angles(Su):
    """
    Return field angles according to Lindegren main eq. 12
    :param Su: array with the proper direction in the SRS reference system
    eta: along-scan field angle
    zeta: across-scan field angle
    """
    if not isinstance(Su, np.ndarray):
        raise TypeError('Su has to be a numpy array, instead is {}'.format(type(Su)))
    if Su.shape != (3,):
        raise ValueError('Shape of Su should be (3), instead it is {}'.format(Su.shape))
    Gamma_c = 0  # angle between the two scanners # TODO: implement gamma_c
    Su_x, Su_y, Su_z = Su[:]

    phi = np.arctan2(Su_y, Su_x)
    if phi >= np.pi or phi < -np.pi:
        raise ValueError('phi should be -pi <= phi < pi, instead it is: {}'.format(phi))
    zeta = np.arctan2(Su_z, np.sqrt(Su_x**2+Su_y**2))

    field_index = np.sign(phi)
    eta = phi - field_index * Gamma_c / 2
    return eta, zeta


def compute_du_dparallax(r, b_G):
    """computes du/dw"""
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
    update = (np.eye(3) - r @ r.T) @ b_G / const.Au_per_Au
    update.shape = (3)  # This way it returns an error if it has to copy data
    return -update  # np.ones(3)  #


def compute_mnu(eta, zeta):
    """
    return column vectors of the S'[m_l, n_l, u_l] matrix
    :param eta: float
    :param zeta: float
    """
    phi = eta  # # WARNING:  implement the correct version (phi != eta)
    # S_mnu = np.zeros((3,3))
    m_l = np.array([-np.sin(phi), np.cos(phi), 0])
    n_l = np.array([-np.sin(zeta)*np.cos(phi), np.sin(zeta)*np.sin(phi), np.cos(zeta)])
    u_l = np.array([np.cos(zeta)*np.cos(phi), np.cos(zeta)*np.sin(phi), np.sin(zeta)])
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
