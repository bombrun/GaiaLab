import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\..')))

import numpy as np
import constants as const
import agis_functions as af
import frame_transformations as ft
from satellite import Satellite
import helpers as helpers



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


def compute_design_equation(true_source,calc_source,observation_times,gaia):
    """
    param true_source : the parameters of the true source
    param calc_source : the parameters of the estimated source
    param observation_times : scanner observation times
    returns : dR_ds_AL, dR_ds_AC, R_AL, R_AC, FA(phi_obs, zeta_obs,phi_calc, zeta_calc)
    """
    alpha0 = calc_source.source.get_parameters()[0]
    delta0 = calc_source.source.get_parameters()[1]
    p0, q0, r0 = ft.compute_pqr(alpha0, delta0)
    n_obs = len(observation_times)
    R_AL = np.zeros(n_obs)
    R_AC = np.zeros(n_obs)
    dR_ds_AL = np.zeros((n_obs, 5))
    dR_ds_AC = np.zeros((n_obs, 5))
    FA = []
    for j, t_l in enumerate(observation_times):
        # one should use the 2 telescopes option for the residuals
        q_l = gaia.func_attitude(t_l)
        phi_obs, zeta_obs = af.observed_field_angles(true_source, q_l, gaia, t_l, True)
        phi_calc, zeta_calc = af.calculated_field_angles(calc_source, q_l, gaia, t_l, True)

        FA.append([phi_obs, zeta_obs,phi_calc, zeta_calc])

        R_AL[j] = (phi_obs-phi_calc)
        R_AC[j] = (zeta_obs-zeta_calc)

        # but not for the derivatives...
        phi_c, zeta_c = af.calculated_field_angles(calc_source, q_l, gaia, t_l, False)
        m, n, u = af.compute_mnu(phi_c, zeta_c)

        du_ds = calc_source.source.compute_du_ds(gaia,p0,q0,r0.reshape(3,1),q_l,t_l)
        dR_ds_AL[j, :] = m @ du_ds.transpose() * helpers.sec(zeta_calc)
        dR_ds_AC[j, :] = n @ du_ds.transpose()
    return dR_ds_AL, dR_ds_AC, R_AL, R_AC, np.array(FA)
