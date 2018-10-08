"""
agis.py
File for the global solutions
Class Agis

author: Luca Zampieri

t (float): time from J2000 [days]
such that t_ep = 0
"""
# # Imports
# Local modules
import frame_transformations as ft
import constants as const
from satellite import Satellite
from source import Source
from agis_functions import *

# global modules
import numpy as np

# TODO: implement all the functions


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


class Agis:

    def __init__(self, sat, sources=[], verbose=False):
        """
        Also contains:
        **Temporary variables**
        self.astro_param : the astronometric parameters for the source we're examining
        self.obs_times : the observation times for a given source
        **Variables**
        """
        self.verbose = verbose
        self.sources = sources
        self.sat = sat
        self.consider_stellar_aberation = False
        num_parameters_per_sources = 5  # the astronomic parameters
        s_vector = np.zeros((len(self.sources)*num_parameters_per_sources, 1))
        self.N_ss = np.zeros((s_vector*np.transpose(s_vector)).shape)
        # Call self.init_blocks()
        # print('The shape of N_ss is {}'.format(N_ss.shape))

        # N_aa =
    def init_blocks(self):
        """
        Initialize the block
        """
        if self.verbose:
            print('initializing N_ss of shape: {}'.format(self.N_ss.shape))
        self.__init_N_ss()
        self.__init_N_aa()

    def __init_N_ss(self):
        """ initialize the matrix N_ss """

        for i in range(0, self.N_ss.shape[0], 5):  # Nss is symmetric and square
            dR_ds_AL, dR_ds_AC = self.dR_ds()
            dR_ds = dR_ds_AC  # TODO: this is a simplicfication, make it real later
            W = np.eye(5)  # TODO: implement the weighting factor
            self.N_ss[i*5:i*5+5, i*5:i*5+5] = dR_ds.transpose() @ dR_ds @ W  # should we use np.sum?
            # The rest of N_ss are zero by initialisation

    def __init_N_aa(self):
        """

        """
        pass


    def compute_source_observations_parameters(self, obs_times, source_num=0):
        """
        Computes parameters corresponding to the observation times
        :param obs_times: [list of floats] list with the observation times
        :param source_num: the index of the source of interest
        """
        if source_num >= len(self.sources):
            raise ValueError('there is no source number {}'.format(source_num))

        self.obs_times = obs_times
        source = self.sources[source_num]
        self.astro_param = np.zeros((len(obs_times), 5))
        for i, t_l in enumerate(obs_times):
            alpha, delta, mu_alpha, mu_delta = source.topocentric_angles(self.sat, t_l)
            parallax = source.parallax
            self.astro_param[i, :] = [alpha, delta, parallax, mu_alpha, mu_delta]

    def update_S_block(self):
        for i, s in enumerate(sources):
            self.update_source(i)

    def update_source(self, i):
        # compute A_i
        # compute h_i
        # compute d_i
        d_i = np.zeros((5, 1))
        return d_i

    def compute_du_ds_tilde(self):
        """
        Compute dũ_ds far a given source
        :param:
        :returns:
        :used names:
            - b_G(t) barycentric position of Gaia at the time of observation, also
              called barycentric ephemeris of the Gaia Satellite
            - t_B barycentric time (takes into account the Römer delay)
        Notes: t_ep in the paper is not used since we assume t_ep=0 and start counting the time from J2000
        """
        # TODO: Consider writing this function with autograd
        # In this function consider all u as being ũ! (for notation we call them here u)
        # Values needed to compute the derivatives
        n_i = len(self.obs_times)  # the number of observations
        source = self.sources[0]  # TODO: make it for every sources

        du_dalpha = np.zeros((n_i, 3))
        du_ddelta = np.zeros((n_i, 3))
        du_dparallax = np.zeros((n_i, 3))
        du_dmualpha = np.zeros((n_i, 3))
        du_dmudelta = np.zeros((n_i, 3))
        # du_ds = np.zeros()

        for j, t_l in enumerate(self.obs_times):
            # t_l being the observation time
            p, q, r = ft.compute_pqr(source.alpha, source.delta)  # alpha delta of this source and observation
            p = np.expand_dims(p, axis=0)
            q = np.expand_dims(q, axis=0)
            r = np.expand_dims(r, axis=0)
            b_G = np.expand_dims(self.sat.ephemeris_bcrs(t_l), axis=0).transpose()
            t_B = t_l + np.dot(r, b_G) / const.c
            tau = t_B - const.t_ep
            Au = 1  # TODO: Check what kind of au we should use, might be constant only in our case

            # Compute derivatives
            du_dalpha[j] = p
            du_ddelta[j] = q
            du_dparallax[j] = ((np.eye(3) - r @ np.transpose(r)) @ b_G / Au).transpose()
            du_dmualpha[j] = p*tau
            du_dmudelta[j] = q*tau

        du_ds = np.array([du_dalpha.transpose(),
                          du_ddelta.transpose(),
                          du_dparallax.transpose(),
                          du_dmualpha.transpose(),
                          du_dmudelta.transpose()])
        print('du_dalpha.shape: {}'.format(du_dalpha.shape))
        print('du_ds.shape: {}'.format(du_ds.shape))
        return du_ds

    def compute_coordinate_direction(self):
        """
        Compute ũ_i(t) which is the coordinate direction. Once taken into account the
        aberration of light this represents the proper direction of the source.
        All in the xyz frame (ICRS)
        """
        source = self.sources[0]

        # TODO: generalise for more sources
        p, q, r = ft.compute_pqr(self.source.alpha, self.source.delta)
        my_vector = r + (t_B - t_ep) * (p * mu_alpha + q * mu_delta + r * mu_r) - parallax * b_G / Au
        my_direction = my_vector/np.norm(my_vector)
        return my_direction

    def coordinates_direction_to_proper_direction(self):
        """take into account aberrationn of light
        :param du_ds_tilde: in the CoRMS frame (lmn)
        :returns du_ds: in the SRS frame (xyz)
        """
        # TODO: implement stellar aberation
        # if self.consider_stellar_aberation:
        # raise ValueError('Stellar aberation not yet implemented')
        # u = compute_coordinate_direction()
        # coeff = (1-u.transpose()*v_g/const.c)I - u*v_G.transpose()/const.c
        coeff = 1
        C_du_ds = coeff * self.compute_du_ds_tilde()
        S_du_ds = self.C_du_ds_to_S_du_ds(C_du_ds)
        if self.verbose:
            print('S_du_ds shape: {}'.format(S_du_ds.shape))
        return S_du_ds

    def du_ds(self):
        """
        returns the derivative of the proper direction w.r.t. the astronomic
        parameters.
        """
        return self.coordinates_direction_to_proper_direction()

    def C_du_ds_to_S_du_ds(self, C_du_ds):
        """
        rotate the frame from CoRMS (lmn) to SRS (xyz) for du_ds
        """
        S_du_ds = np.zeros(C_du_ds.shape)
        for i in range(C_du_ds.shape[0]):  # TODO: remove these ugly for loop
            for j in range(C_du_ds.shape[-1]):
                attitude = self.sat.func_attitude(self.obs_times[j])  # gaia attitude at time t_l
                S_du_ds[i, :, j] = ft.lmn_to_xyz(attitude, C_du_ds[i, :, j])

        return S_du_ds

    def dR_ds_L(self):
        """
        Computes the derivative of the error (R^AL_l) wrt the 5 astronomic parameters
        s_i transposed. Per each observation L
        """

    def dR_ds(self):
        """
        Computes the derivative of the error (R^AL_l) wrt the 5 astronomic parameters
        s_i transposed.
        :param kind: either AL for ALong scan direction or AC for ACross scan direction
        :returns:
        """

        def sec(x):
            """Should be stable since x close to 0"""
            return 1/np.cos(x)
        source = self.sources[0]  # TODO: make it for multiple sources
        du_ds = self.du_ds()
        dR_ds_AL = np.zeros((len(self.obs_times), 5))
        dR_ds_AC = np.zeros(dR_ds_AL.shape)

        for i, t_L in enumerate(self.obs_times):
            eta, xi = compute_field_angles(source, self.sat, t_L)
            m, n, u = compute_mnu(eta, xi)
            dR_ds_AL[i, :] = -m @ du_ds[:, :, i].transpose() * sec(xi)
            dR_ds_AC[i, :] = -n @ du_ds[:, :, i].transpose()

        return dR_ds_AL, dR_ds_AC


if __name__ == '__main__':
    print('Executing agis.py as main file')
