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

# TODO: implement all the functionsto compute condition number of the matrix: numpy.linalg.cond(x)
#


class Calc_source:
    """
    Contains the calculated parameters per source
    """
    def __init__(self, name, obs_times, source_params, mu_radial):
        """ Initial guess of the parameters"""
        self.name = name
        self.obs_times = obs_times  # times at which it has been observed
        self.s_params = source_params  # position at which it has been observed
        self.mu_radial = mu_radial  # not considered an unknown of the problem
        self.s_old = []
        self.errors = []


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
        self.sat = sat
        self.consider_stellar_aberation = False
        self.iter_counter = 0

        num_sources = len(self.real_sources)

        # The four parameter vector
        # self.s_param = np.zeros((num_sources, 5))  # source parameters
        # self.a_param = np.zeros(0)  # attitude parameters
        # self.c_param = np.zeros(0)  # Calibration parameters
        # self.g_param = np.zeros(0)  # Global parameters

        num_parameters_per_sources = 5  # the astronomic parameters
        total_number_of_observations = 0
        for calc_source in self.calc_sources:
            total_number_of_observations += len(calc_source.obs_times)
            calc_source.s_old.append(calc_source.s_params)
        s_vector = np.zeros((len(self.calc_sources)*num_parameters_per_sources, 1))
        self.N_ss = np.zeros((num_sources*5, num_sources*5))  # 5 source params
        self.N_aa = np.zeros((4, 4))  # 4 attitude params  # WARNING: not the correct shpe

        self.init_blocks()

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

    def reset_iterations(self):
        self.init_blocks()
        self.iter_counter = 0
        for calc_source in self.calc_sources:
            calc_source.s_old = []
            calc_source.errors = []

    def error_function(self):
        """
        Compute the error function Q
        """
        error = 0
        for source_index, s in enumerate(self.calc_sources):
            if self.verbose:
                print('source: {}'.format(s.s_params))
            for j, t_L in enumerate(s.obs_times):
                R_L = self.R_L(source_index, t_L)
                error += R_L ** 2
        return error

    def R_L(self, source_index, t):
        """ R = eta_obs + zeta_obs - eta_calc - zeta_calc """
        # WARNING: maybe source is not in the field of vision of sat at time t!
        eta_obs, zeta_obs = observed_field_angles(self.real_sources[source_index],
                                                  self.sat, t)
        obs = eta_obs + zeta_obs
        eta_calc, zeta_calc = compute_field_angles(self.calc_sources[source_index],
                                                   self.real_sources[source_index],
                                                   self.sat, t)
        calc = eta_calc + zeta_calc
        R_L = obs - calc
        return R_L

    def iterate(self, num):
        """
        Do _num_ iterations
        """
        for i in range(num):
            self.iter_counter += 1
            if self.verbose:
                print('***** Iteration: {} *****'.format(self.iter_counter))
                print('Error before iteration: {}'.format(self.error_function()))
            # self.init_blocks()
            self.update_S_block()
            # self.update_A_block()
            if self.verbose:
                print('Error after iteration: {}'.format(self.error_function()))

    def update_S_block(self):
        """ Performs the update of the source parameters """
        for i, calc_source in enumerate(self.calc_sources):
            calc_source.s_old.append(calc_source.s_params.copy())
            calc_source.errors.append(self.error_function())
            self.update_block_S_i(i)

    def update_block_S_i(self, source_index):
        calc_source = self.calc_sources[source_index]
        A = self.block_S_error_rate_matrix(source_index)
        W = np.eye(len(calc_source.obs_times))
        h = self.compute_h(source_index)
        LHS = A.transpose() @ W @ A
        RHS = A.transpose() @ W @ h
        d = np.linalg.solve(LHS, RHS)
        if self.verbose:
            print('dim A: {}'.format(A.shape))
            print('dim W: {}'.format(W.shape))
            print('dim h: {}'.format(h.shape))
            print('dim d: {}'.format(d.flatten().shape))
            print('dim s:', self.calc_sources[source_index].s_params.shape)
        print('d: ', d[3:-1].flatten())
        self.calc_sources[source_index].s_params[:] += d.flatten()

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
        if self.verbose:
            print('du_dalpha.shape: {}'.format(du_dalpha.shape))
            print('du_ds.shape: {}'.format(du_ds.shape))
        return du_ds

    def du_ds(self, calc_source):
        """
        returns the derivative of the proper direction w.r.t. the astronomic
        parameters.
        take into account aberrationn of light
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
                t_L = calc_source.obs_times[j]
                # WARNING: we should not use func_attitude here
                # S_du_ds[i, :, j] = ft.lmn_to_xyz(self.sat.func_attitude(t_L), C_du_ds[i, :, j])
                R = rotation_matrix_from_alpha_delta(self.real_sources[0], self.sat, t_L)
                S_du_ds[i, :, j] = np.array(R@C_du_ds[i, :, j].T)

        return S_du_ds

    def dR_ds(self, source_index):
        """
        Computes the derivative of the error (R_l) wrt the 5 astronomic parameters
        s_i transposed.
        :param kind: either AL for ALong scan direction or AC for ACross scan direction
        :returns:
        """

        def sec(x):
            """ Compute secant. Should be stable since x close to 0"""
            return 1/np.cos(x)

        calc_source = self.calc_sources[source_index]
        du_ds = self.du_ds(calc_source)
        dR_ds_AL = np.zeros((len(calc_source.obs_times), 5))
        dR_ds_AC = np.zeros(dR_ds_AL.shape)

        for i, t_L in enumerate(calc_source.obs_times):
            eta, zeta = compute_field_angles(calc_source, self.real_sources[0], self.sat, i)
            m, n, u = compute_mnu(eta, zeta)
            dR_ds_AL[i, :] = -m @ du_ds[:, :, i].transpose() * sec(zeta)
            dR_ds_AC[i, :] = -n @ du_ds[:, :, i].transpose()

        return dR_ds_AL + dR_ds_AC

    def block_S_error_rate_matrix(self, source_index):
        """error matrix for the block update S"""
        return -self.dR_ds(source_index)

    def compute_h(self, source_index):
        calc_source = self.calc_sources[source_index]
        h = np.zeros((len(calc_source.obs_times), 1))
        for i, t_L in enumerate(calc_source.obs_times):
            h[i, 0] = self.R_L(source_index, t_L)
        if self.verbose:
            print('h: {}'.format(h))
        return h

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
