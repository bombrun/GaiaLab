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


class Calc_source:
    """
    Contains the calculated parameters per source
    """
    def __init__(self, name, obs_times, source_params, attitude_params, mu_radial):
        """ Initial guess of the parameters"""
        self.name = name
        self.obs_times = obs_times  # times at which it has been observerd
        self.a_params = attitude_params  # attitude at which it should be observed
        self.s_params = source_params  # position at which it has been observed
        self.mu_radial = mu_radial  # not considered an unknown of the problem


class Agis:

    def __init__(self, sat, calc_sources=[], real_sources=[], verbose=False):
        """
        Also contains:
        **Temporary variables**
        self.astro_param : the astronometric parameters for the source we're examining
        self.obs_times : the observation times for a given source
        **Variables**
        """
        self.verbose = verbose
        self.calc_sources = calc_sources
        self.real_sources = real_sources
        # self.a_params = self.init_attitude_param()
        self.sat = sat
        self.consider_stellar_aberation = False
        self.iter_counter = 0

        # The four parameter vector
        # self.s = np.zeros(0)  # source parameters
        # self.a = np.zeros(0)  # attitude parameters
        # self.c = np.zeros(0)  # Calibration parameters
        # self.g = np.zeros(0)  # Global parameters

        num_parameters_per_sources = 5  # the astronomic parameters
        total_number_of_observations = 0
        for source in self.calc_sources:
            total_number_of_observations += len(source.obs_times)
        s_vector = np.zeros((len(self.calc_sources)*num_parameters_per_sources, 1))
        self.N_ss = np.zeros((len(self.calc_sources)*5, len(self.calc_sources)*5))  # 5 source params
        self.N_aa = np.zeros((4, 4))  # 4 attitude params
        # Call self.init_blocks()
        # print('The shape of N_ss is {}'.format(N_ss.shape))

    def init_attitude_param(self):
        N = self.sat.s_x.get_coeffs().shape[0]
        a = np.zeros((N, 4))
        # a[:,0] =
        # a[:]

    def init_blocks(self):
        """
        Initialize the block
        """
        if self.verbose:
            print('initializing N_ss of shape: {}'.format(self.N_ss.shape))
            print('initializing N_aa of shape: {}'.format(self.N_aa.shape))
        self.__init_N_ss()
        self.__init_N_aa()

    def __init_N_ss(self):
        """ initialize the matrix N_ss """

        for i in range(0, self.N_ss.shape[0], 5):  # Nss is symmetric and square
            dR_ds = self.dR_ds(i)  # i being the source index
            W = np.eye(5)  # TODO: implement the weighting factor
            self.N_ss[i*5:i*5+5, i*5:i*5+5] = dR_ds.transpose() @ dR_ds @ W  # should we use np.sum?
            # The rest of N_ss are zero by initialisation

    def __init_N_aa(self):
        """
        Initialize the matrix N_aa
        N_aa

        for n in range(0, self.N_aa.shape[0], 4):
        """

        pass

    def compute_source_observations_parameters(self, source_num=0):
        """
        Computes parameters corresponding to the observation times
        :param obs_times: [list of floats] list with the observation times
        :param source_num: the index of the source of interest
        """

        if source_num >= len(self.real_sources):
            raise ValueError('there is no source number {}'.format(source_num))

        obs_times = self.calc_sources[source_num].obs_times
        source = self.real_sources[source_num]
        self.astro_param = np.zeros((len(obs_times), 5))
        for i, t_l in enumerate(obs_times):
            alpha, delta, mu_alpha, mu_delta = source.topocentric_angles(self.sat, t_l)
            parallax = source.parallax
            self.astro_param[i, :] = [alpha, delta, parallax, mu_alpha, mu_delta]

    def error_function(self):
        """
        Compute the error function Q
        """
        error = 0
        for source_index, s in enumerate(self.calc_sources):
            for j, t_L in enumerate(s.obs_times):
                R_L = self.R_L(source_index, j, t_L)
                error += R_L
        return error

    def iterate(self):
        self.iter_counter += 1
        print('***** Iteration: {} *****'.format(self.iter_counter))
        self.init_blocks()
        print('Error before iteration: {}'.format(self.error_function()))
        self.update_S_block()
        # self.update_A_block()
        print('Error after iteration: {}'.format(self.error_function()))

    def update_S_block(self):
        """ Performs the update of the source parameters """
        for i, s in enumerate(self.calc_sources):
            self.update_block_S(i)

    def der_topocentric_function(self, calc_source):
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
        n_i = len(calc_source.obs_times)  # the number of observations

        du_dalpha = np.zeros((n_i, 3))
        du_ddelta = np.zeros((n_i, 3))
        du_dparallax = np.zeros((n_i, 3))
        du_dmualpha = np.zeros((n_i, 3))
        du_dmudelta = np.zeros((n_i, 3))
        # du_ds = np.zeros()

        # For each observation compute du/ds
        for j, t_l in enumerate(calc_source.obs_times):
            # t_l being the observation time
            # using alpha delta of this source and observation
            p, q, r = ft.compute_pqr(calc_source.s_params[0], calc_source.s_params[1])
            p = np.expand_dims(p, axis=0)
            q = np.expand_dims(q, axis=0)
            r = np.expand_dims(r, axis=0)
            b_G = np.expand_dims(self.sat.ephemeris_bcrs(t_l), axis=0).transpose()
            t_B = t_l + np.dot(r, b_G) / const.c
            tau = t_B - const.t_ep
            Au = 1/const.AU_per_pc  # 1  # TODO: Check what kind of au we should use, might be constant only in our case

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

    def compute_der_proper_direction(self, calc_source):
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
        C_du_ds = coeff * self.der_topocentric_function(calc_source)
        S_du_ds = self.C_du_ds_to_S_du_ds(calc_source, C_du_ds)
        if self.verbose:
            print('S_du_ds shape: {}'.format(S_du_ds.shape))
        return S_du_ds

    def C_du_ds_to_S_du_ds(self, calc_source, C_du_ds):
        """
        rotate the frame from CoRMS (lmn) to SRS (xyz) for du_ds
        """
        S_du_ds = np.zeros(C_du_ds.shape)
        for i in range(C_du_ds.shape[0]):  # TODO: remove these ugly for loop
            for j in range(C_du_ds.shape[-1]):
                # gaia attitude at time t_l
                attitude = Quaternion(calc_source.a_params[0], calc_source.a_params[1],
                                      calc_source.a_params[1], calc_source.a_params[3])
                S_du_ds[i, :, j] = ft.lmn_to_xyz(attitude, C_du_ds[i, :, j])

        return S_du_ds

    def du_ds(self, calc_source):
        """
        returns the derivative of the proper direction w.r.t. the astronomic
        parameters.
        """
        return self.compute_der_proper_direction(calc_source)

    def dR_ds(self, source_index):
        """
        Computes the derivative of the error (R_l) wrt the 5 astronomic parameters
        s_i transposed.
        :param kind: either AL for ALong scan direction or AC for ACross scan direction
        :returns:
        """

        def sec(x):
            """Should be stable since x close to 0"""
            return 1/np.cos(x)

        calc_source = self.calc_sources[source_index]
        du_ds = self.du_ds(calc_source)
        dR_ds_AL = np.zeros((len(calc_source.obs_times), 5))
        dR_ds_AC = np.zeros(dR_ds_AL.shape)

        for i, t_L in enumerate(calc_source.obs_times):
            eta, zeta = compute_field_angles(calc_source, self.sat, i)
            m, n, u = compute_mnu(eta, zeta)
            dR_ds_AL[i, :] = -m @ du_ds[:, :, i].transpose() * sec(zeta)
            dR_ds_AC[i, :] = -n @ du_ds[:, :, i].transpose()

        return dR_ds_AL + dR_ds_AC

    def block_S_error_rate_matrix(self, source_index):
        """error matrix for the block update S"""
        return -self.dR_ds(source_index)

    def eta_obs_plus_zeta_obs(self, source, t):
        """ For the moment it returns the exact eta of the defined source"""
        # WARNING: maybe source is not in the field of vision of sat at time t!
        eta, zeta = observed_field_angles(source, self.sat, t)
        return eta + zeta

    def eta_calc_plus_zeta_calc(self, calc_source, i):
        """ For the moment it returns the exact eta of the defined source"""
        # WARNING: maybe source is not in the field of vision of sat at time t!
        eta, zeta = compute_field_angles(calc_source, self.sat, i)
        return eta + zeta

    def R_L(self, source_index, i, t):
        """ R = eta_obs + xi_obs - eta_calc - xi_calc """
        R_L = self.eta_obs_plus_zeta_obs(self.real_sources[source_index], t)
        - self.eta_calc_plus_zeta_calc(self.calc_sources[source_index], i)
        return R_L

    def compute_h(self, source_index):
        calc_source = self.calc_sources[source_index]
        h = np.zeros((len(calc_source.obs_times), 1))
        for i, t_L in enumerate(calc_source.obs_times):
            h[i, 0] = self.R_L(source_index, i, t_L)
        if self.verbose:
            print('h: {}'.format(h))
        return h

    def update_block_S(self, source_index):
        calc_source = self.calc_sources[source_index]
        A = self.block_S_error_rate_matrix(source_index)
        W = np.eye(len(calc_source.obs_times))
        h = self.compute_h(source_index)
        LHS = A.transpose() @ W @ A
        RHS = A.transpose() @ W @ h
        d = np.linalg.solve(LHS, RHS)
        if self.verbose:
            print('dim d: {}'.format(d.flatten().shape))
            print(self.calc_sources[source_index].s_params.shape)
        self.calc_sources[source_index].s_params[:] += d.flatten()

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

    ############################################################################
    # For attitude update
    def dR_dq(self, source_index):
        """compute dR/dq"""

        def sec(x):
            """Should be stable since x close to 0"""
            return 1/np.cos(x)
        calc_source = self.calc_sources[source_index]

        du_dq = self.du_dq(calc_source)
        dR_dq_AL = np.zeros((len(calc_source.obs_times), 5))
        dR_dq_AC = np.zeros(dR_dq_AL.shape)

        for i, t_L in enumerate(calc_source.obs_times):
            eta, zeta = compute_field_angles(calc_source, self.sat, i)
            m, n, u = compute_mnu(eta, zeta)
            # attitude = ??
            dR_dq_AL[i, :] = 2 * sec(zeta) * (attitude * ft.vector_to_quaternion(n)).to_vector()
            dR_dq_AC[i, :] = - 2 * (attitude * ft.vector_to_quaternion(m)).to_vector()

        return dR_dq_AL + dR_dq_AC


if __name__ == '__main__':
    print('Executing agis.py as main file')
