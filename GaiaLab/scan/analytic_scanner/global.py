# # File for the global solutions
#
# Luca Zampieri 2018
# #

# Local modules
import frame_transformations as ft
import constants as const

# global modules
import numpy as np

# TODO: implement all the functions

num_sources = 3
num_parameters_per_sources = 5  # the astronomic parameters
s_vector = np.zeros((num_sources*num_parameters_per_sources, 1))


N_ss = np.zeros((s_vector*np.transpose(s_vector)).shape)
print('The shape of N_ss is {}'.format(N_ss.shape))
for i in range(N_ss.shape[0]):
    for j in range(N_ss.shape[1]):
        # for loop can be removed by looping just over i and saying N_ss[i,i]
        # (i.e. no need for "if" neither)
        if i == j:
            N_ss[i, j] = 0  #
        else:
            N_ss[i, j] = 0
# N_aa =


def eta_obs():
    pass


def f_obs():  # xi_obs():
    pass


def f_calc():
    pass


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


def R_l(s, a, c, g):
    return f_obs() - f_calc()


def coordinates_direction_to_proper_direction():
    """take into account aberrationn of light
    :param du_ds_tilde:
    :returns du_ds:
    """
    coeff = 1
    return coeff * du_ds_tilde()


def du_ds_tilde(source, satellite, observation_times):
    """
    Compute dũ_ds far a given source
    :param:
    :returns:
    :used names:
        - b_G(t) barycentric position of Gaia at the time of observation, also
          called barycentric ephemeris of the Gaia Satellite
        - t_B barycentric time (takes into account the Römer delay)
    """
    # TODO: Consider writing this function with autograd
    # In this function consider all u as being ũ! (for notation we call them here u)
    # Values needed to compute the derivatives

    du_dalpha = np.zeros(())
    du_ddelta =
    du_dparallax =
    du_dmualpha = p*tau
    du_dmudelta = q*tau

    for j, t_l in enumerate(observation_times):
        # t_l being the observation time
        p, q, r = ft.compute_pqr(source.alpha, source.delta)  # alpha delta of this source and observation
        p[j] = p
        q[j] = q
        r[j] = r

        t_B = t_l + np.transpose(r) * satellite.ephemeris_bcrs(t_l) / const.c
        tau = t_B - t_ep  # TODO: define t_ep!
    # Compute derivatives
    du_dalpha = p
    du_ddelta = q
    du_dparallax = (np.eye(3) - r * np.transpose(r)) * b_G(t_l) / A_u
    du_dmualpha = p*tau
    du_dmudelta = q*tau

    du_ds = np.array([du_dalpha,
                      du_ddelta,
                      du_dparallax,
                      du_dmualpha,
                      du_dmudelta])
    return du_ds


def dR_ds(kind='AL'):
    """
    Computes the derivative of the error (R^AL_l) wrt the 5 astronomic parameters
    s_i transposed.
    :param kind: either AL for ALong scan direction or AC for ACross scan direction
    :returns:
    """
    def sec(x):
        return 1/np.cos(x)

    if kind == 'AL':
        dR_ds = -m*du_ds()*sec(xi)  # TODO: dont forget the quaternion multiplication
    elif kind == 'AC':
        dR_ds = -n*du_ds()  # TODO:  dont forget the quaternion multiplication p18
    else:
        raise ValueError('parameter "kind" can be either "AL" or "AC", but not {}'.format(kind))
    return dR_ds


class AGIS:

    def __init__(self, sources=[]):
        self.sources = sources

    def update_S_block(self):
        for i, s in enumerate(sources):
            self.update_source(i)

    def update_source(self, i):
        # compute A_i
        # compute h_i
        # compute d_i
        d_i = np.zeros((5, 1))
        return d_i