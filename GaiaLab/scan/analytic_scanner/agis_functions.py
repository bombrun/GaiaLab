# -*- coding: utf-8 -*-

"""
File agis_helpers.py
functions that uses the classes source, satellite but don't belong to a
given file yet
:used by: (at least) agis.py & scanner.py
:author: LucaZampieri

When cleaning this file search for ???, LUCa, warning , error, debug, print?

*Notes:*
    In this file, when there is a reference, unless explicitly stated otherwise,
    it refers to Lindegren main article:
    "The astronometric core solution for the Gaia mission - overview of models,
    algorithms, and software implementation" by L. Lindegren, U. Lammer, D. Hobbs,
    W. O'Mullane, U. Bastian, and J.Hernandez
    The reference is usually made in the following way: Ref. Paper eq. [1]

TODO:
    - [DONE] Rotate the attitude
    - [DONE] attitude i.e. generate observations (with scanner object but without scanning)
    - [DONE] with scanner
    - [DONE] two telescope
    - Attitute with scanner
    - scaling

Other:
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
import quaternion
# Local imports
import constants as const
import helpers as helpers
import frame_transformations as ft
from source import Source
from satellite import Satellite
from source import compute_topocentric_direction


def generate_observation_wrt_attitude(attitude):
    """
    returns right ascention and declination corresponding to the direction in
    which the x-vector rotated to *attitude* is pointing
    returns alpha, delta in radians
    """
    artificial_Su = [1, 0, 0]
    artificial_Cu = ft.xyz_to_lmn(attitude, artificial_Su)
    alpha, delta = ft.vector_to_alpha_delta(artificial_Cu)
    return alpha, delta


def error_between_func_attitudes(my_times, func_att1, func_att2):
    error_in_attitude = 0
    for t in my_times:
        diff_att = 0
        att1 = func_att1(t)
        att2 = func_att2(t)
        diff_att += np.abs(att2.w - att1.w)
        diff_att += np.abs(att2.x - att1.x)
        diff_att += np.abs(att2.y - att1.y)
        diff_att += np.abs(att2.z - att1.z)
        error_in_attitude += np.abs(diff_att)
    return error_in_attitude


def rotation_matrix_from_alpha_delta(source, sat, t):
    Cu = source.unit_topocentric_function(sat, t)
    Su = np.array([1, 0, 0])
    r = helpers.get_rotation_matrix(Cu, Su)
    return r


def attitude_from_alpha_delta(source, sat, t, vertical_angle_dev=0):
    """:param vertical_angle_dev: how much we deviate from zeta"""
    Cu = source.unit_topocentric_function(sat, t)
    Su = np.array([1, 0, 0])
    if vertical_angle_dev == 0:
        vector, angle = helpers.get_rotation_vector_and_angle(Cu, Su)
        q_out = quaternion.from_rotation_vector(angle*vector)
    else:
        Cu_xy = helpers.normalize(np.array([Cu[0], Cu[1], 0]))  # Cu on S-[xy] plane
        v1, a1 = helpers.get_rotation_vector_and_angle(Cu_xy, Su)
        q1 = quaternion.from_rotation_vector(v1*a1)

        Su_xy = ft.rotate_by_quaternion(q1.inverse(), Su)  # Su rotated to be on same xy than Cu_xy
        v2, a2 = helpers.get_rotation_vector_and_angle(Cu, Su_xy)
        q2_dev = quaternion.from_rotation_vector(v2*(a2+vertical_angle_dev))
        # deviaetd_Su = ft.rotate_by_quaternion(q2_dev.inverse(), Su_xy)
        q_out = q1*q2_dev
        # angle -= 0.2
    return q_out


def spin_axis_from_alpha_delta(source, sat, t):
    Cu = source.unit_topocentric_function(sat, t)
    Su = np.array([1, 0, 0])
    vector, angle = helpers.get_rotation_vector_and_angle(Cu, Su)
    # vector = vector/np.linalg.norm(vector)
    # satellite_position = sat.ephemeris_bcrs(t)
    return vector


def scanning_y_coordinate(source, sat, t):
    raise ValueError('This function is obsolete')
    # raise ValueError('Check that ')
    att = get_fake_attitude(source, sat, t)
    y_vec = ft.rotate_by_quaternion(att, [0, 1, 0])
    # vector = vector/np.linalg.norm(vector)
    # satellite_position = sat.ephemeris_bcrs(t)
    return y_vec


def get_fake_attitude(source, sat, t):
    quat1 = attitude_from_alpha_delta(source, sat, t)
    # quat2 = Quaternion(vector=np.array([1, 0, 0]), angle=const.sat_angle)
    attitude = quat1  # * quat2
    return attitude  # sat.func_attitude(t)


def get_angular_FFoV_PFoV(sat, t):
    """
    return angular positions (alpha, delta) of the fields of view as a
    function of time.
    """
    z_axis = np.array([0, 0, 1])
    attitude = sat.func_attitude(t)

    quat_PFoV = quaternion.from_rotation_vector(z_axis*const.Gamma_c / 2)
    quat_FFoV = quaternion.from_rotation_vector(z_axis*(-const.Gamma_c / 2))

    PFoV_SRS = ft.rotate_by_quaternion(quat_PFoV, np.array([1, 0, 0]))
    FFoV_SRS = ft.rotate_by_quaternion(quat_FFoV, np.array([1, 0, 0]))

    PFoV_CoMRS = ft.xyz_to_lmn(attitude, PFoV_SRS)
    FFoV_CoMRS = ft.xyz_to_lmn(attitude, FFoV_SRS)

    alpha_PFoV, delta_PFoV = ft.vector_to_alpha_delta(PFoV_CoMRS)
    alpha_FFoV, delta_FFoV = ft.vector_to_alpha_delta(FFoV_CoMRS)

    return alpha_PFoV, delta_PFoV, alpha_FFoV, delta_FFoV


# ### For scanner --------------------------------------------------------------
def get_interesting_days(ti, tf, sat, source, zeta_limit):
    # print(zeta_limit)
    day_list = []
    zeta_limit = min(zeta_limit*6, 3)  # why *6 ?? [rad]
    time_step = 1
    days = np.arange(ti, tf, time_step)
    for t in days:
        attitude = sat.func_attitude(t)
        eta, zeta = observed_field_angles(source, attitude, sat, t)
        if np.abs(zeta) < zeta_limit:
            day_list.append(t)

    return day_list


def generate_scanned_times_intervals(day_list, time_step):
    extend_by = 1
    previous_days = list(np.array(day_list)-extend_by)
    extended_days = set(day_list + previous_days)
    scanned_intervals = []
    for day in day_list:
        scanned_intervals += list(np.arange(day, day+extend_by, time_step))
    scanned_intervals = list(set(scanned_intervals))
    return scanned_intervals


# ### End for scanner ##########################################################


# ### For attitude updating: ---------------------------------------------------
# ## Just for plotting
def compare_attitudes(gaia, Solver, my_times):
    fig = plt.figure()
    colors = ['red', 'orange', 'blue', 'green']
    labels_gaia = ["w_gaia", "x_gaia", "y_gaia", "z_gaia"]
    labels_solver = ["w_solv", "x_solv", "y_solv", "z_solv"]
    gaia_attitudes = [gaia.s_w(my_times), gaia.s_x(my_times),
                      gaia.s_y(my_times), gaia.s_z(my_times)]
    solver_attitudes = []
    for i in range(4):
        plt.plot(my_times, gaia_attitudes[i], ':', color=colors[i], label=labels_gaia[i])
        plt.plot(my_times, Solver.attitude_splines[i](my_times), '--', color=colors[i], label=labels_solver[i])
    plt.xlabel("my_times [%s]" % len(my_times))
    plt.legend(loc=9, bbox_to_anchor=(1.1, 1))
    plt.title('Attitudes in time intervals')
    plt.show()


def multi_compare_attitudes(gaia, Solver, my_times):
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    # axes = [axs[0, 0], axs[0, 1], axs[ 1,0], axs[1,1]]
    colors = ['red', 'cyan', 'blue', 'green']
    titles = ["w", "x", "y", "z"]
    labels_gaia = ["w_gaia", "x_gaia", "y_gaia", "z_gaia"]
    labels_solver = ["w_solv", "x_solv", "y_solv", "z_solv"]
    gaia_attitudes = [gaia.s_w(my_times), gaia.s_x(my_times),
                      gaia.s_y(my_times), gaia.s_z(my_times)]
    solver_attitudes = []
    error_component = []
    for i, ax in enumerate(axs):

        Solver_attitude = Solver.attitude_splines[i](my_times)
        error_component = np.abs(gaia_attitudes[i] - Solver_attitude).sum()
        ax.plot(my_times, gaia_attitudes[i], '.:', color='k', label=labels_gaia[i])
        ax.plot(my_times, Solver_attitude, ':+', color=colors[i], label=labels_solver[i],
                alpha=0.8)
        ax.set_title(titles[i] + ' error: ' + str(error_component))
        ax.grid(), ax.legend(), ax.set_xlabel("my_times [%s]" % len(my_times))

    plt.suptitle('Attitudes in time intervals')
    plt.show()
    return fig


def multi_compare_attitudes_errors(gaia, Solver, my_times):
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    # axes = [axs[0, 0], axs[0, 1], axs[ 1,0], axs[1,1]]
    colors = ['red', 'cyan', 'blue', 'green']
    titles = ["w-error", "x-error", "y-error", "z-error"]
    labels_gaia = ["w_gaia", "x_gaia", "y_gaia", "z_gaia"]
    labels_solver = ["w_solv", "x_solv", "y_solv", "z_solv"]
    gaia_attitudes = [gaia.s_w(my_times), gaia.s_x(my_times),
                      gaia.s_y(my_times), gaia.s_z(my_times)]
    solver_attitudes = []
    error_component = []
    for i, ax in enumerate(axs):
        Solver_attitude = Solver.attitude_splines[i](my_times)
        error_component = np.abs(gaia_attitudes[i] - Solver_attitude)
        total_error = error_component.mean()
        ax.plot(my_times, error_component, ':', color=colors[i],
                label='diff |' + labels_gaia[i] + '-' + labels_solver[i] + '|')
        ax.set_title(titles[i] + ': ' + str(total_error))
        ax.grid(), ax.legend(), ax.set_xlabel("my_times [%s]" % len(my_times))

    plt.suptitle('Attitudes in time intervals')
    plt.show()
    return fig
# ## end just for plotting


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
    att_coeffs, att_splines = ([], [])
    internal_knots = attitude_splines[0].get_knots()  # chose [0] since all the same
    att_knots = extend_knots(internal_knots, k)  # extend the knots to have all the needed ones
    for i, spline in enumerate(attitude_splines):
        coeffs = spline.get_coeffs()
        att_coeffs.append(coeffs)
        att_splines.append(BSpline(att_knots, coeffs, k))
    return np.array(att_coeffs), np.array(att_knots),  np.array(att_splines)


def get_times_in_knot_interval(time_array, knots, index, M):
    """
    :param time_array: [numpy array]
    :return: times in knot interval defined by [index, index+M]
    """
    return time_array[(knots[index] < time_array) & (time_array < knots[index+M])]


def get_left_index(knots, t, M):
    """
    :param M: spline order (k+1)
    :returns left_index: the left_index corresponding to t i.e. *i* s.t.
        $t_i < t < t_{i+1}$
    """
    left_index_array = np.where(knots <= t)
    if not list(left_index_array[0]):
        raise ValueError('t smaller than smallest knot')
    left_index = left_index_array[0][-1]
    return left_index


def extend_knots(internal_knots, k):
    extended_knots = []
    for i in range(k):
        extended_knots.append(internal_knots[0])
    extended_knots += list(internal_knots)
    for i in range(k):
        extended_knots.append(internal_knots[-1])
    return extended_knots


def compute_coeff_basis_sum(coeffs, bases, L, M, time_index):
    """
    Computes the sum:

    .. math::
        \sum_{n=L-M+1}^{L}(a_n \cdot b_n)

    :param coeffs: [numpy array] splines coefficients
    :param bases: [numpy array] B-spline bases
    :param L: [int] left_index
    :param M: [int] spline order (= spline degree + 1)
    :param time_index: [float] time index where we want to evaluate the spline
    :returns: [numpy array] vector of the
    """
    # Note the +1 to include last term
    return np.sum(bases[L-M+1:L+1, time_index] * coeffs[:, L-M+1:L+1], axis=1)


def compute_attitude_deviation(coeff_basis_sum):
    """
    :Action: Compute the attitude deviation from unity
    :param coeff_basis_sum: the sum(a_n*b_n) with n=L-M+1 : L
    :returns: attitude deviation from unity D_l"""
    return 1 - np.linalg.norm(coeff_basis_sum)**2


def compute_DL_da_i(coeff_basis_sum, bases, time_index, i):
    """
    Compute derivative of the attitude deviation wrt attitude params
    :param coeff_basis_sum: the sum(a_n*b_n) with n=L-M+1 : L
    """
    dDL_da = -2 * coeff_basis_sum * bases[i, time_index]
    return dDL_da.reshape(4, 1)


def compute_DL_da_i_from_attitude(attitude, bases, time_index, i):
    """
    Ref. Paper eq. [83]
    Compute derivative of the attitude deviation wrt attitude params
    """
    dDL_da = -2 * quaternion.as_float_array(attitude) * bases[i, time_index]
    return dDL_da.reshape(4, 1)


def compute_dR_dq(calc_source, sat, attitude, t):
    """ Ref. Paper eq. [79]
    return [array] with dR/dq"""
    # Here below we have "phi" since we set double_telescope to False
    phi, zeta = calculated_field_angles(calc_source, attitude, sat, t, double_telescope=False)
    Sm, Sn, Su = compute_mnu(phi, zeta)
    q = attitude

    dR_dq_AL = 2 * helpers.sec(zeta) * (q * ft.vector_to_quat(Sn))
    dR_dq_AC = -2 * (q * ft.vector_to_quat(Sm))

    return (quaternion.as_float_array(dR_dq_AL),  quaternion.as_float_array(dR_dq_AC))


def dR_da_i(dR_dq, bases_i):
    """ :param basis_i: B-spline basis of index i"""
    dR_da_i = dR_dq * bases_i
    return dR_da_i.reshape(4, 1)
# ### End attitude updating ####################################################


# ### Beginning field angles and associated functions --------------------------
def observed_field_angles(source, attitude, sat, t, double_telescope=False):
    """
    Ref. Paper eq. [12]-[13]
    Return field angles according to Lindegren eq. 12
    eta: along-scan field angle (== phi if double_telescope = False)
    zeta: across-scan field angle
    """
    Cu = source.unit_topocentric_function(sat, t)  # u in CoMRS frame
    Su = ft.lmn_to_xyz(attitude, Cu)
    # if double_telescope is False, it will return (phi, zeta)
    eta, zeta = compute_field_angles(Su, double_telescope)
    return eta, zeta


def calculated_field_angles(calc_source, attitude, sat, t, double_telescope=False):
    """
    Ref. Paper eq. [12]-[13]
    Return field angles according to Lindegren eq. 12
    eta: along-scan field angle
    """
    alpha, delta, parallax, mu_alpha, mu_delta = calc_source.s_params[:]
    params = np.array([alpha, delta, parallax, mu_alpha, mu_delta, calc_source.mu_radial])

    Cu = compute_topocentric_direction(params, sat, t)  # u in CoMRS frame
    Su = ft.lmn_to_xyz(attitude, Cu)  # u in SRS frame

    eta, zeta = compute_field_angles(Su, double_telescope)
    return eta, zeta


def compute_field_angles(Su, double_telescope=False):
    """
    Ref. Paper eq. [12]-[13]
    Return field angles according to ref. Paper eq. [12]
    :param Su: array with the proper direction in the SRS reference system
    eta: along-scan field angle
    zeta: across-scan field angle
    """
    if not isinstance(Su, np.ndarray):
        raise TypeError('Su has to be a numpy array, instead is {}'.format(type(Su)))
    if Su.shape != (3,):
        raise ValueError('Shape of Su should be (3), instead it is {}'.format(Su.shape))
    if double_telescope:
        Gamma_c = const.Gamma_c  # angle between the two scanners # TODO: implement gamma_c
    else:
        Gamma_c = 0
    Su_x, Su_y, Su_z = Su[:]

    phi = np.arctan2(Su_y, Su_x)
    if phi >= np.pi or phi < -np.pi:
        raise ValueError('phi should be -pi <= phi < pi, instead it is: {}'.format(phi))
    zeta = np.arctan2(Su_z, np.sqrt(Su_x**2+Su_y**2))

    field_index = np.sign(phi)
    eta = phi - field_index * Gamma_c / 2
    return eta, zeta


def compute_mnu(phi, zeta):
    """
    Ref. Paper eq. [69]
    return column vectors of the S'[m_l, n_l, u_l] matrix
    :param phi: float
    :param zeta: float
    """
    m_l = np.array([-np.sin(phi), np.cos(phi), 0])
    n_l = np.array([-np.sin(zeta)*np.cos(phi), np.sin(zeta)*np.sin(phi), np.cos(zeta)])
    u_l = np.array([np.cos(zeta)*np.cos(phi), np.cos(zeta)*np.sin(phi), np.sin(zeta)])
    return np.array([m_l, n_l, u_l])
# ### End field angles and associated functions ################################


# ### For source updating: -----------------------------------------------------
def compute_du_dparallax(r, b_G):
    """Ref. Paper eq. [73]
    computes du/dw"""
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
    update = -(np.eye(3) - r @ r.T) @ b_G / const.Au_per_Au
    update.shape = (3)  # This way it returns an error if it has to copy data
    return update  # np.ones(3)  #
# ###End source updating #######################################################


# ### Beginning Color aberration -----------------------------------------------
def compute_deviated_angles_color_aberration(eta, zeta, color, error):
    parameter = 1/10
    if error != 0:
        eta = eta + parameter * color
        zeta = zeta + parameter * color
    return eta, zeta
# ### End Color aberration #####################################################


# End of file
