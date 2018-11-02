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
from scipy.interpolate import BSpline


# TODO: implement all the functions to compute condition number of the matrix: numpy.linalg.cond(x)
#


class Calc_source:
    """
    Contains the calculated parameters per source
    """
    def __init__(self, name, obs_times, source_params, mu_radial):
        """
        Initial guess of the parameters
        :source_params: alpha, delta, parallax, mu_alpha, mu_delta
        """
        self.name = name
        self.obs_times = obs_times  # times at which it has been observed
        self.s_params = source_params  # position at which it has been observed
        self.mu_radial = mu_radial  # not considered an unknown of the problem
        self.s_old = []
        self.errors = []


class Agis:

    def __init__(self, sat, calc_sources=[], real_sources=[], attitude_splines=None,
                 verbose=False, spline_order=4, attitude_regularisation_factor=1e1):
        """
        Also contains:
        **Temporary variables**
        self.astro_param : the astronometric parameters for the source we're examining
        self.obs_times : the observation times for a given source
        **Variables**
        # The four parameter vector
        # self.s_param = np.zeros((num_sources, 5))  # source parameters
        # self.a_param = np.zeros(0)  # attitude parameters
        # self.c_param = np.zeros(0)  # Calibration parameters
        # self.g_param = np.zeros(0)  # Global parameters
        """
        self.k = spline_order  # spline order
        self.M = self.k + 1
        self.attitude_regularisation_factor = attitude_regularisation_factor
        self.verbose = verbose
        self.calc_sources = calc_sources
        self.real_sources = real_sources
        self.sat = sat
        self.consider_stellar_aberation = False
        self.iter_counter = 0

        num_sources = len(self.real_sources)

        num_parameters_per_sources = 5  # the astronomic parameters
        all_obs_times = []
        self.time_dict = {}
        for source_index, calc_source in enumerate(self.calc_sources):
            all_obs_times += list(calc_source.obs_times)
            for t in calc_source.obs_times:
                self.time_dict[t] = source_index
            calc_source.s_old.append(calc_source.s_params)
        s_vector = np.zeros((len(self.calc_sources)*num_parameters_per_sources, 1))
        self.all_obs_times = np.sort(all_obs_times)
        total_number_of_observations = len(self.all_obs_times)

        if attitude_splines is not None:  # Set everything for the attitude
            c, t, ks, s = extract_coeffs_knots_from_splines(attitude_splines, self.k)
            self.att_coeffs, self.att_knots, self.attitude_splines = (c, t, s)
            # for k in ks:
            #    print('k:', k)
            self.set_splines_basis()  # set the basis Bsplines

    def reset_iterations(self):
        self.iter_counter = 0
        for calc_source in self.calc_sources:
            calc_source.s_old = []
            calc_source.errors = []

    def get_quaternion_COMRS_2_SRS(self, t, source_index=0):
        """Get quaternion that rotates from CoMRS to SRS"""
        # WARNING: when taking the attitude from the satellite the quaternion should be inverted!!!
        quat = attitude_from_alpha_delta(self.real_source[source_index], self.sat, t)
        return quat

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
        R_eta = 0
        R_zeta = 0
        eta_obs, zeta_obs = observed_field_angles(self.real_sources[source_index],
                                                  self.sat, t)
        attitude = attitude_from_alpha_delta(self.real_sources[source_index], self.sat, t)
        eta_calc, zeta_calc = calculated_field_angles(self.calc_sources[source_index],
                                                      attitude,
                                                      self.sat, t)
        R_eta = eta_obs - eta_calc  # AL
        R_zeta = zeta_obs - zeta_calc  # AC
        R_L = R_eta + R_zeta
        return R_L

    def iterate(self, num):
        """
        Do _num_ iterations
        """
        for i in range(num):
            self.iter_counter += 1
            print('***** Iteration: {} *****'.format(self.iter_counter))
            if self.verbose:
                print('Error before iteration: {}'.format(self.error_function()))
            # self.init_blocks()
            # self.update_S_block()
            self.update_A_block()
            error = error_between_func_attitudes(self.all_obs_times, self.sat.func_attitude, self.get_attitude)
            print(error)
            if self.verbose:
                print('Error after iteration: {}'.format(self.error_function()))

    def update_S_block(self):
        """ Performs the update of the source parameters """
        for i, calc_source in enumerate(self.calc_sources):
            calc_source.s_old.append(calc_source.s_params.copy())
            calc_source.errors.append(self.error_function())
            self.update_block_S_i(i)
            print('parallax: ', self.calc_sources[i].s_params[2])

    def update_block_S_i(self, source_index):
        """update source #i"""
        calc_source = self.calc_sources[source_index]
        A = self.block_S_error_rate_matrix(source_index)
        W = np.eye(len(calc_source.obs_times))
        h = self.compute_h(source_index)
        LHS = A.transpose() @ W @ A
        RHS = A.transpose() @ W @ h
        d = np.linalg.solve(LHS, RHS)
        d = d.flatten()
        self.calc_sources[source_index].s_params[:] += d
        if self.verbose:
            print('dim A:{}\ndim W:{}\ndim h:{}\ndim d:{}'
                  .format(A.shape, W.shape, h.shape, d.shape))

    def du_tilde_ds(self, source_index):
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
        calc_source = self.calc_sources[source_index]
        n_i = len(calc_source.obs_times)  # the number of observations
        du_ds = np.zeros((5, 3, n_i))
        alpha = calc_source.s_params[0]  # + calc_source.s_params[3]*t_l
        delta = calc_source.s_params[1]  # + calc_source.s_params[4]*t_l
        p, q, r = ft.compute_pqr(alpha, delta)
        r.shape = (3, 1)  # reshapes r
        # For each observation compute du/ds
        for j, t_l in enumerate(calc_source.obs_times):  # t_l being the observation time
            b_G = self.sat.ephemeris_bcrs(t_l)
            t_B = t_l  # + np.dot(r, b_G) / const.c
            tau = t_B - const.t_ep
            # Compute derivatives
            du_dalpha = p
            du_ddelta = q
            du_dparallax = compute_du_dparallax(r, b_G)
            du_dmualpha = p*tau
            du_dmudelta = q*tau
            du_ds[:, :, j] = [du_dalpha, du_ddelta, du_dparallax, du_dmualpha, du_dmudelta]
        if self.verbose:
            print('du_ds.shape: {}'.format(du_ds.shape))
        return du_ds

    def du_ds(self, source_index):
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
        calc_source = self.calc_sources[source_index]
        coeff = 1
        C_du_ds = coeff * self.du_tilde_ds(source_index)
        S_du_ds = self.C_du_ds_to_S_du_ds(source_index, C_du_ds)
        if self.verbose:
            print('S_du_ds shape: {}'.format(S_du_ds.shape))
        return S_du_ds

    def C_du_ds_to_S_du_ds(self, source_index, C_du_ds):
        """
        rotate the frame from CoRMS (lmn) to SRS (xyz) for du_ds
        """
        calc_source = self.calc_sources[source_index]
        S_du_ds = np.zeros(C_du_ds.shape)
        for i in range(C_du_ds.shape[0]):  # TODO: remove these ugly for loop
            for j in range(C_du_ds.shape[-1]):
                t_L = calc_source.obs_times[j]
                # WARNING: we should use quaternion object!
                # S_du_ds[i, :, j] = ft.lmn_to_xyz(self.sat.func_attitude(t_L), C_du_ds[i, :, j])
                # R = rotation_matrix_from_alpha_delta(self.real_sources[source_index], self.sat, t_L)
                # S_du_ds[i, :, j] = np.array(R@C_du_ds[i, :, j].T)
                attitude = attitude_from_alpha_delta(self.real_sources[source_index], self.sat, t_L)
                S_du_ds[i, :, j] = ft.rotate_by_quaternion(attitude, C_du_ds[i, :, j])
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
        du_ds = self.du_ds(source_index)
        dR_ds_AL = np.zeros((len(calc_source.obs_times), 5))
        dR_ds_AC = np.zeros(dR_ds_AL.shape)

        for i, t_L in enumerate(calc_source.obs_times):
            attitude = attitude_from_alpha_delta(self.real_sources[source_index], self.sat, t_L)
            eta, zeta = calculated_field_angles(calc_source, attitude, self.sat, i)
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

    ############################################################################
    # For attitude update
    def get_attitude(self, t):
        s_w = self.attitude_splines[0]
        s_x = self.attitude_splines[1]
        s_y = self.attitude_splines[2]
        s_z = self.attitude_splines[3]
        return Quaternion(s_w(t), s_x(t), s_y(t), s_z(t)).unit()

    def create_attitude(self, t):  # # WARNING:
        pass

    def set_splines_basis(self):
        """Set the Bsplines bases function into self.att_bases"""
        bases = []
        for i in range(self.att_coeffs.shape[0]):  # for each attitude parameter
            knots = self.att_knots[i]
            coeffs = self.att_coeffs[i]
            bases.append(get_basis_Bsplines(knots, coeffs, self.k, self.all_obs_times))
        self.att_bases = np.array(bases)

    def actualise_splines(self):
        for i in range(self.attitude_splines.shape[0]):
            self.attitude_splines[i] = BSpline(self.att_knots[i], self.att_coeffs[i], k=self.k)

    def update_A_block(self):
        LHS = self.compute_attitude_LHS()
        RHS = self.compute_attitude_RHS()
        d = np.linalg.solve(LHS, RHS)
        c_update = d.reshape(4, -1)
        self.att_coeffs += c_update
        for i in range(self.att_coeffs.shape[1]):
            self.att_coeffs[:, i] /= np.linalg.norm(self.att_coeffs[:, i])
        self.actualise_splines()  # Create the new splines

    def compute_attitude_LHS(self):
        N_aa_dim = self.att_coeffs.shape[1]  # *4
        print(N_aa_dim)
        N_aa = np.zeros((N_aa_dim*4, N_aa_dim*4))
        for n in range(0, N_aa_dim):
            for m in range(0, N_aa_dim):
            # for m in range(max(n-self.M+1, 0), min(n+self.M-1, N_aa_dim)):
                N_aa[n*4:n*4+4, m*4:m*4+4] = self.compute_dR_da_mn(m, n)
        return N_aa

    def compute_attitude_RHS(self):
        N_aa_dim = self.att_coeffs.shape[1]
        RHS = np.zeros((N_aa_dim*4, 1))
        for n in range(0, N_aa_dim):
            RHS[n*4:n*4+4] = self.compute_attitude_RHS_n(n)
        return RHS

    def get_source_index(self, t):
        """ get the index of the source corresponding to observation t"""
        if t in self.time_dict:
            return self.time_dict[t]
        else:
            raise ValueError('time not in time_dict')

    def compute_attitude_RHS_n(self, n_index):
        rhs = np.zeros((4, 1))
        # WARNING: here we take the knots of w since they should be the same for the 4 components
        observed_times = get_times_in_knot_interval(self.all_obs_times, self.att_knots[0], n_index, self.M)
        for i, t_L in enumerate(observed_times):
            source_index = self.get_source_index(t_L)
            calc_source = self.calc_sources[source_index]
            attitude = self.get_attitude(t_L)
            left_index = get_left_index(self.att_knots[0], t_L, M=self.M)
            obs_time_index = list(self.all_obs_times).index(t_L)

            # Compute the regulation part
            coeff_basis_sum = compute_coeff_basis_sum(self.att_coeffs, self.att_bases,
                                                      left_index, self.M, obs_time_index)
            D_L = compute_attitude_deviation(coeff_basis_sum)
            # dDL_da_n = compute_DL_da_i(coeff_basis_sum, self.att_bases, obs_time_index, n_index)
            dDL_da_n = compute_DL_da_i_from_attitude(attitude, self.att_bases, obs_time_index, n_index)
            regularisation_part = self.attitude_regularisation_factor**2 * dDL_da_n * D_L

            # # WARNING: Here we put the Across scan and the along scan together
            dR_dq = compute_dR_dq(calc_source, self.sat, attitude, t_L)
            dR_da_n = dR_da_i(dR_dq, self.att_bases[:, n_index, obs_time_index])
            R_L = self.compute_attitude_R_L(source_index, attitude, t_L)
            rhs += dR_da_n * R_L + regularisation_part.reshape(4, -1)

        return -rhs

    def compute_attitude_R_L(self, source_index, attitude, t):
        """ R = eta_obs + zeta_obs - eta_calc - zeta_calc """
        # WARNING: maybe source is not in the field of vision of sat at time t!
        R_eta = 0
        R_zeta = 0
        eta_obs, zeta_obs = observed_field_angles(self.real_sources[source_index],
                                                  self.sat, t)
        eta_calc, zeta_calc = calculated_field_angles(self.calc_sources[source_index],
                                                      attitude,
                                                      self.sat, t)
        R_eta = eta_obs - eta_calc  # AL
        R_zeta = zeta_obs - zeta_calc  # AC
        R_L = R_eta + R_zeta
        return R_L

    def compute_dR_da_mn(self, m_index, n_index):
        """compute dR/da (i.e. wrt coeffs)"""
        dR_da_mn = np.zeros((4, 4))
        # WARNING: here we take the knots of w since they should be the same for the 4 components
        observed_times_m = get_times_in_knot_interval(self.all_obs_times, self.att_knots[0], m_index, self.M)
        observed_times_n = get_times_in_knot_interval(self.all_obs_times, self.att_knots[0], n_index, self.M)
        observed_times_mn = list(set(observed_times_m) & set(observed_times_n))  # intersection of the lists

        for i, t_L in enumerate(observed_times_mn):
            source_index = self.get_source_index(t_L)
            calc_source = self.calc_sources[source_index]
            attitude = self.get_attitude(t_L)
            left_index = get_left_index(self.att_knots[0], t_L, M=self.M)
            obs_time_index = list(self.all_obs_times).index(t_L)

            # Compute the regulation part
            coeff_basis_sum = compute_coeff_basis_sum(self.att_coeffs, self.att_bases,
                                                      left_index, self.M, obs_time_index)
            # dDL_da_n = compute_DL_da_i(coeff_basis_sum, self.att_bases, obs_time_index, n_index)
            # dDL_da_m = compute_DL_da_i(coeff_basis_sum, self.att_bases, obs_time_index, m_index)
            dDL_da_n = compute_DL_da_i_from_attitude(attitude, self.att_bases, obs_time_index, n_index)
            dDL_da_m = compute_DL_da_i_from_attitude(attitude, self.att_bases, obs_time_index, m_index)
            regularisation_part = self.attitude_regularisation_factor**2 * dDL_da_n @ dDL_da_m

            # Ccompute the original objective function part
            # # WARNING: Here we put the Across scan and the along scan together
            dR_dq = compute_dR_dq(calc_source, self.sat, attitude, t_L)
            dR_da_m = dR_da_i(dR_dq, self.att_bases[:, m_index, obs_time_index])
            dR_da_n = dR_da_i(dR_dq, self.att_bases[:, n_index, obs_time_index])
            dR_da_mn += dR_da_n @ dR_da_m.T + regularisation_part
            if m_index < 0:
                if i < 2:
                    print('**** m:', m_index, '**** n:', n_index, '**** i:', i)
                    print('dR_dq: ', dR_dq)
                    print('dR_da_m', dR_da_m)
                    print('dR_da_n', dR_da_n)
                    print('dR_da_mn', dR_da_mn)

        return dR_da_mn


if __name__ == '__main__':
    print('Executing agis.py as main file')
