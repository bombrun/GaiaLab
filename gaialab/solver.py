import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\..')))

import numpy as np
import constants as const
import agis_functions as af
import frame_transformations as ft
from satellite import Satellite
import helpers as helpers
from agis import Calc_source
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
    dR_ds_AL = np.zeros((n_obs, 5))
    dR_ds_AC = np.zeros((n_obs, 5))
    FA = []
    for j, t_l in enumerate(observation_times):
        # fake attitude using the position of the true sources at the given time
        # i.e. not based on the nominal scanning law
        q_l = af.attitude_from_alpha_delta(true_source,gaia,t_l,0)
        phi_obs, zeta_obs = af.observed_field_angles(true_source, q_l,  gaia, t_l, False)
        phi_calc, zeta_calc = af.calculated_field_angles(calc_source, q_l, gaia, t_l, False)

        FA.append([phi_obs, zeta_obs,phi_calc, zeta_calc])

        R_AL[j] = (phi_obs-phi_calc)
        R_AC[j] = (zeta_obs-zeta_calc)

        m, n, u = af.compute_mnu(phi_calc, zeta_calc)

        du_ds = true_source.compute_du_ds(gaia,p,q,r,q_l,t_l)
        dR_ds_AL[j, :] = m @ du_ds.transpose() * helpers.sec(zeta_calc)
        dR_ds_AC[j, :] = n @ du_ds.transpose()
    print ("ok")


    return dR_ds_AL, dR_ds_AC, R_AL, R_AC, np.array(FA)
