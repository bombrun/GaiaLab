"""
Solver implementation in Python

:Author:MaraBucur
"""
import numpy as np
import quaternion
import constants as const
import frame_transformations as ft
from satellite import Satellite
import helpers as helpers
from source import Calc_source
from source import Source

def solve_AL(true_source,calc_source,observation_times):
    """
    perform one step of the source solver using only along scan observations
    """
    # get the design equation
    dR_ds_AL, dR_ds_AC, R_AL, R_AC, FA = compute_design_equation(true_source,calc_source,observation_times)
    # build the normal equation
    N = dR_ds_AL.transpose() @ dR_ds_AL
    rhs = dR_ds_AL.transpose() @ R_AL
    # solve the normal equation
    updates = np.linalg.solve(N,rhs)
    # update the calculated source parameters
    # take care of alpha
    calc_source.s_params[0] = calc_source.s_params[0] + updates[0] * np.cos(calc_source.s_params[1])
    calc_source.s_params[1:] = calc_source.s_params[1:] + updates[1:]

def compute_design_equation(true_source,calc_source,gaia,observation_times):
    """
    param true_source : the parameters of the true source
    param calc_source : the parameters of the estimated source
    param observation_times : a list of times that will be used to create observation
        (they do not necessarly correspond to a realistic scanning law,
        indeed the true attitude is taken using the position of the true source at these times)
    returns : dR_ds_AL, dR_ds_AC, R_AL, R_AC, FA(phi_obs, zeta_obs,phi_calc, zeta_calc)
    """
    gaia=Satellite(0,365*5,1/24)
    alpha0 = calc_source.source.get_parameters()[0]
    delta0 = calc_source.source.get_parameters()[1]
    p, q, r = ft.compute_pqr(alpha0, delta0)
    n_obs = len(observation_times)
    R_AL = np.zeros(n_obs)
    R_AC = np.zeros(n_obs)
    dR_ds_AL = np.zeros((n_obs, 7))
    dR_ds_AC = np.zeros((n_obs, 7))
    FA = []
    for j, t_l in enumerate(observation_times):
        # fake attitude using the position of the true sources at the given time
        # i.e. not based on the nominal scanning law
        q_l = attitude_from_alpha_delta(true_source,gaia,t_l,0)
        phi_obs, zeta_obs = field_angles(true_source, q_l,  gaia, t_l, False)
        phi_calc, zeta_calc =field_angles(calc_source, q_l, gaia, t_l, False)

        FA.append([phi_obs, zeta_obs,phi_calc, zeta_calc])

        R_AL[j] = (phi_obs-phi_calc)
        R_AC[j] = (zeta_obs-zeta_calc)

        m, n, u = compute_mnu(phi_calc, zeta_calc)

        du_ds = true_source.compute_du_ds(gaia,p,q,r,q_l,t_l)
        dR_ds_AL[j, :] = m @ du_ds.transpose() * helpers.sec(zeta_calc)
        dR_ds_AC[j, :] = n @ du_ds.transpose()



    return dR_ds_AL, dR_ds_AC, R_AL, R_AC, np.array(FA)

def compute_field_angles(Su, double_telescope=False):
    """
    | Ref. Paper [LUDW2011]_ eq. [12]-[13]
    | Return field angles according to eq. [12]

    :param Su: array with the proper direction in the SRS reference system
    :param double_telescope: [bool] If true, uses the model with two telescopes
    :returns:
        * eta: along-scan field angle (== phi if double_telescope = False)
        * zeta: across-scan field angle
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

def attitude_from_alpha_delta(source, sat, t, vertical_angle_dev=0):
    """
    :param source: [Source object]
    :param sat: [satellite object]
    :param t: [float] time
    :param vertical_angle_dev: how much we deviate from zeta
    """
    Cu = source.compute_u(sat, t)
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


def compute_mnu(phi, zeta):
    """
    | Ref. Paper [LUDW2011]_ eq. [69]
    | :math:`S'm_l=[-sin(\phi_l), cos(\phi_l), 0]^T`
    | :math:`S'n_l=[-sin(\zeta_l)cos(\phi_l), -sin(\zeta_l)\cos(\phi_l), cos(\zeta_l)]^T`
    | :math:`S'u_l=[cos(\zeta_l)cos(\phi_l), cos(\zeta_l)sin(\phi_l), sin(\zeta_l)]^T`

    :param phi: [float]
    :param zeta: [float]
    :returns: [array] column vectors of the S'[m_l, n_l, u_l] matrix
    """
    m_l = np.array([-np.sin(phi), np.cos(phi), 0])
    n_l = np.array([-np.sin(zeta)*np.cos(phi), np.sin(zeta)*np.sin(phi), np.cos(zeta)])
    u_l = np.array([np.cos(zeta)*np.cos(phi), np.cos(zeta)*np.sin(phi), np.sin(zeta)])
    return np.array([m_l, n_l, u_l])


def field_angles(calc_source, attitude, sat, t, double_telescope=False):
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
    #alpha, delta, parallax, mu_alpha, mu_delta, g_alpha, g_delta, mu_radial= calc_source.source.get_parameters()
    #params = np.array([alpha, delta, parallax, mu_alpha, mu_delta, g_alpha, g_delta, mu_radial ])

    Cu = calc_source.compute_u(sat, t)  # u in CoMRS frame
    Su = ft.lmn_to_xyz(attitude, Cu)  # u in SRS frame

    eta, zeta = compute_field_angles(Su, double_telescope)
    return eta, zeta
