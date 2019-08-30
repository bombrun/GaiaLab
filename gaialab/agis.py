# -*- coding: utf-8 -*-
"""
:File: agis.py

:purpose: Contains implementation of classes Calc_source and Agis

:Authors: LucaZampieri

.. note:
    - In this file, when there is a reference, unless explicitly stated otherwise,
    it refers to Lindegren main article:
    "The astronometric core solution for the Gaia mission - overview of models,
    algorithms, and software implementation" by L. Lindegren, U. Lammer,
    D. Hobbs, W. O'Mullane, U. Bastian, and J.Hernandez
    The reference is usually made in the following way; Ref. Paper [LUDW2011]_ eq. [1]
    - t (float),  time from J2000 [days]
    such that t_ep = 0

Description of what we are trying to achieve in this file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It is a simplification of what is in [LUDW2011]_ .
**If in doubt, check with the paper.**

The goal would be to minimize:

.. math::
    min_{s,a} Q = \sum_{l \\in AL} (R^{AL}_l)^2
        + \sum_{l \\in AC}  (R^{AC_l})^2

In the following we might forget about :math:`R^{AL}, R^{AC}` and just use
:math:`R_l` (for simplicity)

The necessary condition for optimality would be that the derivative is null. So
given that we want to optimize with respect to variable **x** we would have:

.. math::
    \\frac{dQ}{dx} = \sum_l 2 R_l \\frac{dR_l}{dx} = 0

Later we will see that x will correspond to the source parameters and the
coefficient describing the attitude. (i.e. :math:`x \in \{s, a\}`, **s** being
the the set of 5 source parameter and a being the coeffeicients of the splines
for each of the four quaternions parameters describing the attitude of the
satellite at each time)

This condition being sufficient if Q is convex.

Q being non-linear we are gonna linearize it. Assuming our initial guess is good
enough we can write the optimal set of parameters
:math:`x^* = x^{ref} + x`

We would get:

.. math::
    R_l(x^*) = R_l(x^{ref}) + \\frac{dR_l(x^{ref})}{dx}(x^*-x^{ref})
    = R_l(x^{ref}) + \\frac{dR(x^{ref})}{dx}  x

    Q(x)=[R_l(x^{ref})^2 + 2 R_l(x^{ref}) \\frac{dR(x^{ref})}{dx}x
          + (\\frac{dR_l(x^{ref}) }{dx}x)^2]

Thus at optimality we should have (:math:`\\frac{dQ}{dx}=0`):

.. math::
    [\\frac{dR_l(x^{ref})}{dx}]^T [\\frac{dR_l(x^{ref})}{dx}] x
    = - \\frac{dR_l(x^{ref})}{dx} R_l(x^{ref})

.. note::
    Maily for next contributors:

    - Chowleski factorization (:math:`LL^T`) is about twice as efficient as LU
      factorization (default used by numpy.linalg.solve/scipy) if the matrix is
      hermitian which is the case for us (we have a symmetric matrix)

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
from scipy import sparse as sps
import quaternion  # moble's quaternion (numpy compatible quaternions)


class Calc_source:
    """
    Contains the calculated parameters per source
    """
    def __init__(self,source, name=None, obs_times=[]):
        """
        Data structure containing our computed parameters for the source in
        question.

        :param name: [string] the name of the source
        :param obs_times: [list or array] of the observed times for this source
        :param source_params: [list or array] alpha, delta, parallax, mu_alpha, mu_delta
        :param mu_radial: [float] radial velocity of the source (appart since we
         do not solve for radial velocity)
        :param mean_color: [float] mean color of the source if we want to use
         explore chromatic aberration
        :param source: [source] instead of most of the above parameters we can
         provide a source object instead and take the data from it.
         Manually providing the parameters will override the source parameter

        see :class:`source.Source`

        >>> calc_source = Calc_source('calc_sirio', [1, 2.45, 12], [1, 2, 3, 4, 5], 6)
        >>> calc_source = Calc_source(obs_times=[1, 2, 3], source=sirio)  # where sirio is a source object

        """
        self.source=source
        self.obs_times = obs_times  # times at which it has been observed
        self.s_old = self.source.get_parameters()
        self.errors = []


    def set_params(self, params):
        self.s_old = self.source.get_parameters()
        self.source.set_params(params)


    def compute_u(self,sat,t):
        return self.source.compute_u(sat,t)

class Agis:

    def __init__(self, sat, calc_sources=[], real_sources=[], attitude_splines=None,
                 verbose=False, spline_degree=3, attitude_regularisation_factor=0,
                 updating='attitude', degree_error=0, double_telescope=False):
        """
        :param sat: [Satellite]
        :param calc_sources: [list of Calc_sources]
        :param real_sources: [liste of Sources]
        :param attitude_splines: list of splines, only if we update attitude
        :param verbose: [bool] if True prints intermediate results
        :param spline_degree: degree of the splines
        :param attitude_regularization_factor: [float] lambda in the paper.
         Describes how much we take into account the regularization into our
         minimization problem
        :param updating: [string] defines what we are currently updating
         can be: `'source', 'scanned source' or 'attitude'`
        :param degree_error:
        :param double_telescope: [bool] if True uses two telescopes (gaia-like)
        """
        # Raise an error if invalid initialization
        if len(real_sources) != len(calc_sources):
                raise ValueError('there must be the same number of real and calc sources')
        # Objects
        #: List of the sources objects
        self.real_sources = real_sources
        #: List of calculated sources with 1-1 correspondance to the real sources
        self.calc_sources = calc_sources
        #: Satellite object that we are using to solve the problem
        self.sat = sat

        # Constants
        #: Degree of the interpolating polynomial
        self.k = spline_degree
        #: Order of the spline (degree+1)
        self.M = self.k + 1
        self.attitude_regularisation_factor = attitude_regularisation_factor
        self.verbose = verbose
        self.updating = updating
        self.consider_stellar_aberation = False  # TODO: remove because obsolete?
        self.degree_error = degree_error  # [only for source] deviation in vertical direction of the attitude
        self.double_telescope = double_telescope  # bool indicating if we use the double_telescope config

        # Mutable:
        self.iter_counter = 0
        self.N = 0  # not necessary
        self.attitude_der_matrix = None  # sparse attitude derivative matrix
        self.attitude_reg_matrix = None  # sparse attitude derivative matrix
        self.discretized_attitude = None  # attitude evaluated at all the observed times

        # Setting observation times
        all_obs_times = []
        #: [dict] containig the observed times assiciated with their source index
        self.time_dict = {}
        for source_index, calc_source in enumerate(self.calc_sources):
            all_obs_times += list(calc_source.obs_times)
            for t in calc_source.obs_times:
                self.time_dict[t] = source_index
        #: [array] containing the combined observation time of all the sources
        self.all_obs_times = np.sort(all_obs_times)

        # Set attitude
        if attitude_splines is not None:  # Set everything for the attitude
            c, t, s = extract_coeffs_knots_from_splines(attitude_splines, self.k)
            self.c = c.copy()
            self.att_coeffs, self.att_knots, self.attitude_splines = (c.copy(), t, s)
            self.att_bases = get_basis_Bsplines(self.att_knots, self.att_coeffs[0], self.k, self.all_obs_times)
            self.N = self.att_coeffs.shape[1]  # number of coeffs per parameter

            if (self.att_knots.shape[0] - self.att_coeffs.shape[1]) != self.M:
                    raise ValueError('there should be M coeffs less than the number of knots')

    # ### Generic functions for all kind of updating -----------------------------------
    def reset_iterations(self):
        """
        Resetting the iteration counter
        """
        print('Not resetting everything! Call again the solver instead')
        self.iter_counter = 0
        for calc_source in self.calc_sources:
            calc_source.s_old = []
            calc_source.errors = []

    def error_function(self):
        """
        | Ref. Paper [LUDW2011]_ eq. [24]
        | Compute the error function Q

        .. math:: Q = \sum_{l \\in AL} (R^{AL})^2
            + \sum_{l \\in AC}  (R^{AC})^2

        """
        error = 0
        for source_index, s in enumerate(self.calc_sources):
            if self.verbose:
                print('source: {}'.format(s.s_params))
            for j, t_L in enumerate(s.obs_times):
                R_L_AL, R_L_AC = self.compute_R_L(source_index, t_L)
                error += (R_L_AL ** 2 + R_L_AC ** 2)
        return error / self.all_obs_times.shape[0]  # /const.rad_per_mas

    def get_field_angles(self, source_index, t):
        """
        Uses functions :meth:`get_attitude_for_source`, :meth:`get_attitude`,
        :meth:`observed_field_angles`, :meth:`calculated_field_angles`,
        :meth:`satellite.Satellite.func_attitude`

        :param source_index: [int] index of the source
        :param t: [float] [days] time at which
        :returns: eta_obs, zeta_obs, eta_calc, zeta_calc
        """
        # Set attitude, it depends if we wanna update only sources or also attitude params
        if self.updating == 'source':
            attitude = self.get_attitude_for_source(source_index, t)
            attitude_gaia = attitude
        elif self.updating == 'scanned source':
            attitude = self.sat.func_attitude(t)
            attitude_gaia = attitude
        elif self.updating == 'attitude':
            attitude = self.get_attitude(t)
            attitude_gaia = self.sat.func_attitude(t)
        else:
            raise ValueError('incorrect value for self.updating')

        eta_obs, zeta_obs = observed_field_angles(self.real_sources[source_index],
                                                  attitude_gaia,
                                                  self.sat, t, self.double_telescope)
        eta_calc, zeta_calc = calculated_field_angles(self.calc_sources[source_index],
                                                      attitude,
                                                      self.sat, t, self.double_telescope)
        return eta_obs, zeta_obs, eta_calc, zeta_calc

    def deviate_field_angles_color_aberration(self, source_index, t, angles):
        """
        | apply color aberration deviation to field angles (eta, zeta)

        .. note:: If we the color aberration is not given in the sources,
            nothing happens and the same angles are returned.

        """
        # # WARNING: check also deviation in the source update
        eta_obs, zeta_obs, eta_calc, zeta_calc = angles
        return eta_obs, zeta_obs, eta_calc, zeta_calc

    def compute_R_L(self, source_index, t):
        """
        | Ref. Paper [LUDW2011]_ eq. [25]-[26].
        | :math:`R = eta_{obs} + zeta_{obs} - eta_{calc} - zeta_{calc}`

        :note: R_AL = R_eta, R_AC = R_zeta
        :returns: [tuple] R_eta, R_zeta
        """
        # WARNING: maybe source is not in the field of vision of sat at time t!
        R_L_eta, R_L_zeta = (0, 0)

        angles = self.get_field_angles(source_index, t)

        eta_obs, zeta_obs, eta_calc, zeta_calc = self.deviate_field_angles_color_aberration(source_index, t, angles)

        R_L_eta = eta_obs - eta_calc  # AL
        R_L_zeta = zeta_obs - zeta_calc  # AC

        return (R_L_eta, R_L_zeta)

    def iterate(self, num, use_sparse=False, verbosity=0):
        """
        Do _num_ iterations
        """
        if self.verbose is True:
            verbosity += 1

        for i in range(num):
            self.iter_counter += 1
            if verbosity > 0:
                print('***** Iteration: {} *****'.format(self.iter_counter))
                if verbosity > 1:
                    print('Error before iteration: {}'.format(self.error_function()))

            if self.updating == 'source' or self.updating == 'scanned source':
                self.update_S_block()

            elif self.updating == 'attitude':
                self.update_A_block(use_sparse)
                error = error_between_func_attitudes(self.all_obs_times, self.sat.func_attitude, self.get_attitude)
                if verbosity > 1:
                    print('attitude error:', error)
            if verbosity > 0:
                print('Error after iteration: {}'.format(self.error_function()))

    # ### End generic functions ################################################

    #
    # ### Functions only for sources updating (source and scanned source)
    def update_S_block(self):
        """ Performs the update of the source parameters """
        for i, calc_source in enumerate(self.calc_sources):
            calc_source.s_old.append(calc_source.s_params.copy())
            calc_source.errors.append(self.error_function())
            self.update_block_S_i(i)

    def update_block_S_i(self, source_index):
        """
        | Ref. Paper [LUDW2011]_ eq. [57]
        | Uses function :meth:`block_S_error_rate_matrix`

        :param source_index: [int] Index of the source that will be updated
        :action: update source number *source_index*
        """
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

    def compute_h(self, source_index):
        """
        | Ref. Paper [LUDW2011]_ eq. [59]
        | Source update Right hand side
        | Uses functions :meth:`compute_R_L`

        .. warning:: here we sum along-scan and across-scan observations. It
            should be good for this update, but be carefull in future
            implementations

        :param source_index: [int] Index of the source that will be updated
        """
        calc_source = self.calc_sources[source_index]
        h = np.zeros((len(calc_source.obs_times), 1))
        for i, t_L in enumerate(calc_source.obs_times):
            R_L_AL, R_L_AC = self.compute_R_L(source_index, t_L)
            h[i, 0] = (R_L_AL + R_L_AC)
        return h

    def block_S_error_rate_matrix(self, source_index):
        """
        | Ref. Paper [LUDW2011]_ eq. [58]
        | error matrix for the block update S

        uses :meth:`compute_du_ds` and :meth:`dR_ds`

        :param source_index: [int] Index of the source that will be updated
        """
        du_ds = self.compute_du_ds(source_index)
        dR_ds_AL, dR_ds_AC = self.dR_ds(source_index, du_ds)
        return - (dR_ds_AL + dR_ds_AC)

    def dR_ds(self, source_index, du_ds):
        """
        Ref. Paper eq. [71]
        Computes the derivative of the error (R_l) wrt the 5 astronomic parameters
        s_i transposed.

        :param source_index: [int] Index of the source that will be updated
        :param du_ds: [numpy array] derivative of the topocentric function wrt
                      astronometric parameters
        :returns: [tuple of numpy arrays] with derivatives of observations in
                  the Along-scan (AL) and Across-scan (AC) directions
        """

        calc_source = self.calc_sources[source_index]
        dR_ds_AL = np.zeros((len(calc_source.obs_times), 5))
        dR_ds_AC = np.zeros(dR_ds_AL.shape)

        # Iterate through the observation times of the source we are currently
        # updating
        for i, t_L in enumerate(calc_source.obs_times):
            if self.updating == 'source':
                attitude = self.get_attitude_for_source(source_index, t_L)
            elif self.updating == 'scanned source':
                attitude = self.sat.func_attitude(t_L)
            else:
                raise ValueError('not yet implemented for this kind of updating')
            # Set double_telescope to False to get phi
            phi, zeta = calculated_field_angles(calc_source, attitude, self.sat, i, double_telescope=False)
            #color not working
            # phi, zeta = compute_deviated_angles_color_aberration(phi, zeta, calc_source.mean_color, self.degree_error)
            m, n, u = compute_mnu(phi, zeta)
            dR_ds_AL[i, :] = -m @ du_ds[:, :, i].transpose() * helpers.sec(zeta)
            dR_ds_AC[i, :] = -n @ du_ds[:, :, i].transpose()

        return (dR_ds_AL, dR_ds_AC)

    def compute_du_ds(self, source_index):
        """
        | Ref. Paper [LUDW2011]_ eq. [73]
        | Compute dũ_ds for a given source

        .. note::
            - b_G(t) barycentric position of Gaia at the time of observation, also
              called barycentric ephemeris of the Gaia Satellite
            - t_B barycentric time (takes into account the Römer delay)
            - t_ep in the paper is not used since we assume t_ep=0 and start
              counting the time from J2000

        Uses :meth:`CoMRS_to_SRS_for_source_derivatives`,
        :meth:`satellite.Satellite.ephemeris_bcrs` and
        :meth:`agis_functions.compute_du_dparallax`

        :param source_index: [int] Index of the source that will be updated
        :returns: du_ds - source derivatives (wrt astrometric parameters) in SRS
        """
        # In this function consider all u as being ũ! (for notation we call them here u)
        # Values needed to compute the derivatives
        calc_source = self.calc_sources[source_index]
        n_i = len(calc_source.obs_times)  # the number of observations
        du_ds = np.zeros((5, 3, n_i))
        alpha = calc_source.s_params[0]
        delta = calc_source.s_params[1]
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
            CoMRS_derivatives = [du_dalpha, du_ddelta, du_dparallax, du_dmualpha, du_dmudelta]
            SRS_derivatives = self.CoMRS_to_SRS_for_source_derivatives(CoMRS_derivatives, calc_source,
                                                                       t_l, source_index)
            du_ds[:, :, j] = SRS_derivatives
        return du_ds

    def CoMRS_to_SRS_for_source_derivatives(self, CoMRS_derivatives, calc_source, t_L, source_index):
        """
        | Ref. Paper [LUDW2011]_ eq. [72]
        | rotate the frame from CoRMS (lmn) to SRS (xyz) for the given derivatives

        :param CoMRS_derivatives: [list] of deriatives [du_dalpha, du_ddelta,
         du_dparallax, du_dmualpha, du_dmudelta]
        :param calc_source: [Calc_source]
        :param t_L: [float] time of observation
        :param source_index: [int]
        """
        SRS_derivatives = []
        if self.updating == 'source':
            attitude = self.get_attitude_for_source(source_index, t_L)
        elif self.updating == 'scanned source':
            attitude = self.sat.func_attitude(t_L)
        else:  # attitude = self.get_attitude(t_L)
            raise ValueError('Not yet implemented for this case')

        for derivative in CoMRS_derivatives:  # TODO: remove these ugly for loop
            SRS_derivatives.append(ft.lmn_to_xyz(attitude, derivative))
        return SRS_derivatives

    def get_attitude_for_source(self, source_index, t):
        """ For only source updating with **color aberration**.
        Change if condition to decide which sources are affected by that aberration

        :param source_index: [int] Index of the source that will be updated
        :param t: [float] [days] time at which we want the attitude
        """
        if source_index < 0:
            deviation = self.degree_error * const.rad_per_deg  # number in degrees and converted in radians
        else:
            deviation = 0
        return attitude_from_alpha_delta(self.real_sources[source_index], self.sat, t, deviation)
    # ### End function only for source updating ################################

    #
    # ### For attitude update --------------------------------------------------
    def get_attitude(self, t, unit=True):
        """
        Get attitude from the attitude coefficients at time *t*. If *unit*
        is True, the return normalized attitude.

        :param t: [float] time at which we desire the attitude
        :param unit: [bool] if true normalize the quaternion
        :returns: [Quaternion object] attitude
        """
        s_w = self.attitude_splines[0]
        s_x = self.attitude_splines[1]
        s_y = self.attitude_splines[2]
        s_z = self.attitude_splines[3]
        attitude = np.quaternion(s_w(t), s_x(t), s_y(t), s_z(t))
        if unit:
            attitude = attitude.normalized()
        return attitude

    def actualise_splines(self):
        """
        :action: Update the splines re-creating them from the new coefficients
        """
        for i in range(self.attitude_splines.shape[0]):
            self.attitude_splines[i] = BSpline(self.att_knots, self.att_coeffs[i], k=self.k)

    def update_A_block(self, use_sparse=False):  # one
        """
        | Ref. Paper [LUDW2011]_ Section 5.2
        | Solve for the attitude
        """
        if use_sparse is True:
            der_band, reg_band = self.compute_attitude_banded_derivative_and_regularisation_matrices()
            self.compute_sparses_matrices(der_band, reg_band)
            LHS = self.attitude_der_matrix + self.attitude_reg_matrix
            RHS = self.compute_attitude_RHS()
            d = sps.linalg.spsolve(LHS, RHS)

        else:
            LHS = self.compute_attitude_LHS()
            # LHS = self.N_aa
            RHS = self.compute_attitude_RHS()
            # RHS = self.h
            d = np.linalg.solve(LHS, RHS)
            # L = np.linalg.cholesky(LHS)
            # y = np.linalg.solve(L, RHS)
            # d = np.linalg.solve(L.T, y)
            # d = np.linalg.lstsq(LHS, RHS)  # not what it is for
        self.d = d.reshape(self.att_coeffs.shape)
        c_update = d.reshape(self.att_coeffs.shape)
        for i in range(0, self.att_coeffs.shape[1]):
            c_update[0, i] = d[i*4]
            c_update[1, i] = d[i*4+1]
            c_update[2, i] = d[i*4+2]
            c_update[3, i] = d[i*4+3]
        self.c_update = c_update.copy()
        self.att_coeffs[:, :] += c_update[:, :].copy()
        self.actualise_splines()  # Create the new splines

    def update_A_block_bis(self):
        """
        | updates the four components separately (wrong but not by much)
        | It should work if we update just some of the blocks (like one or two)
        """
        LHS = self.compute_attitude_LHS()
        RHS = self.compute_attitude_RHS()
        for i in range(4):
            d = np.linalg.solve(LHS[i::4, i::4], RHS[i::4])
            c_update = d.reshape(-1)
            self.att_coeffs[i] += c_update
        self.actualise_splines()  # Create the new splines

    def compute_attitude_LHS(self):
        """
        Ref. Paper [LUDW2011]_ eq. [82]
        """
        N_aa_dim = self.att_coeffs.shape[1]  # *4
        N_aa = np.zeros((N_aa_dim*4, N_aa_dim*4))
        for n in range(0, N_aa_dim):  # # TODO:  take advantage of the symmetry
            for m in range(0, N_aa_dim):  # # TODO: avoid doing the brute force version
                # for m in range(max((n-self.k), 0), min((n+self.k)+1, N_aa_dim+1)):
                N_aa[n*4:n*4+4, m*4:m*4+4] = self.compute_Naa_mn(m, n)
        self.N_aa = N_aa
        return N_aa

    def compute_attitude_RHS(self):
        """
        Ref. Paper [LUDW2011]_ eq. [82]
        """
        N_aa_dim = self.att_coeffs.shape[1]
        RHS = np.zeros((N_aa_dim*4, 1))
        for n in range(0, N_aa_dim):
            RHS[n*4:n*4+4] = self.compute_attitude_RHS_n(n)
        self.h = RHS.copy()
        return RHS

    def get_source_index(self, t):
        """
        get the index of the source corresponding to observation time t

        :param t: [float] observation time
        :return: [int] source index
        """
        if t in self.time_dict:
            return self.time_dict[t]
        else:
            raise ValueError('time not in time_dict')

    def compute_attitude_RHS_n(self, n_index):
        """
        Ref. Paper [LUDW2011]_ eq. [82]

        :param n_index:
        :return: entry **n** of the Right Hand Side column vector for the
         attitude update
        """
        rhs = np.zeros((4, 1))
        time_support_spline_n = get_times_in_knot_interval(self.all_obs_times, self.att_knots, n_index, self.M)

        # Iterate through the support of spline_n
        for i, t_L in enumerate(time_support_spline_n):
            source_index = self.get_source_index(t_L)
            calc_source = self.calc_sources[source_index]

            attitude = self.get_attitude(t_L, unit=False)
            left_index = get_left_index(self.att_knots, t_L, M=self.M)
            obs_time_index = list(self.all_obs_times).index(t_L)

            # Get the regulation part:
            coeff_basis_sum = compute_coeff_basis_sum(self.att_coeffs, self.att_bases,
                                                      left_index, self.M, obs_time_index)
            D_L = compute_attitude_deviation(coeff_basis_sum)
            dDL_da_n = compute_DL_da_i(coeff_basis_sum, self.att_bases, obs_time_index, n_index)
            # dDL_da_n = compute_DL_da_i_from_attitude(attitude, self.att_bases, obs_time_index, n_index)
            regularisation_part = self.attitude_regularisation_factor**2 * dDL_da_n * D_L

            # Get derivatives:
            dR_dq_AL, dR_dq_AC = compute_dR_dq(calc_source, self.sat, attitude, t_L)
            dR_da_n_AL = dR_da_i(dR_dq_AL, self.att_bases[n_index, obs_time_index])
            dR_da_n_AC = dR_da_i(dR_dq_AC, self.att_bases[n_index, obs_time_index])

            R_L_AL, R_L_AC = self.compute_R_L(source_index, t_L)

            rhs += regularisation_part + dR_da_n_AL * R_L_AL + dR_da_n_AC * R_L_AC
        return -rhs

    def compute_Naa_mn(self, m_index, n_index):
        """
        | Ref. Paper [LUDW2011]_ eq. [82]
        | compute dR/da (i.e. wrt coeffs)

        :param m_index:
        :param n_index:
        :return: entry [n, m] of the attitude normal matrix N_a
        """
        Naa_mn = np.zeros((4, 4))

        # Get times in the support of both spline_m and spline_n
        time_support_spline_m = get_times_in_knot_interval(self.all_obs_times, self.att_knots, m_index, self.M)
        time_support_spline_n = get_times_in_knot_interval(self.all_obs_times, self.att_knots, n_index, self.M)
        time_support_spline_mn = np.sort(helpers.get_lists_intersection(time_support_spline_m, time_support_spline_n))

        # Iterate through all observation in the support of spline n and spline m
        for i, t_L in enumerate(time_support_spline_mn):
            # for i, t_L in enumerate(self.all_obs_times):
            calc_source = self.calc_sources[self.get_source_index(t_L)]
            attitude = self.get_attitude(t_L, unit=False)
            left_index = get_left_index(self.att_knots, t=t_L, M=self.M)
            obs_time_index = list(self.all_obs_times).index(t_L)

            # Compute the regulation part
            coeff_basis_sum = compute_coeff_basis_sum(self.att_coeffs, self.att_bases,
                                                      left_index, self.M, obs_time_index)
            dDL_da_n = compute_DL_da_i(coeff_basis_sum, self.att_bases, obs_time_index, n_index)
            dDL_da_m = compute_DL_da_i(coeff_basis_sum, self.att_bases, obs_time_index, m_index)
            # dDL_da_n = compute_DL_da_i_from_attitude(attitude, self.att_bases, obs_time_index, n_index)
            # dDL_da_m = compute_DL_da_i_from_attitude(attitude, self.att_bases, obs_time_index, m_index)
            regularisation_part = self.attitude_regularisation_factor**2 * dDL_da_n @ dDL_da_m.T

            # Compute the original objective function part
            dR_dq_AL, dR_dq_AC = compute_dR_dq(calc_source, self.sat, attitude, t_L)

            dR_da_m_AL = dR_da_i(dR_dq_AL, self.att_bases[m_index, obs_time_index])
            dR_da_m_AC = dR_da_i(dR_dq_AC, self.att_bases[m_index, obs_time_index])

            dR_da_n_AL = dR_da_i(dR_dq_AL, self.att_bases[n_index, obs_time_index])
            dR_da_n_AC = dR_da_i(dR_dq_AC, self.att_bases[n_index, obs_time_index])

            Naa_mn += regularisation_part + dR_da_n_AL @ dR_da_m_AL.T + dR_da_n_AC @ dR_da_m_AC.T

        return Naa_mn

    # ### Sparse implementation of attitude update-----
    def compute_attitude_banded_derivative_and_regularisation_matrices(self):
        """
        | Ref. Paper [LUDW2011]_ eq. [82]
        | Computes the bands of the banded matrices of the derivatives for the
          normal matrix and the band for the regularization of the normal matrix.

        :return: (der_band, reg_band)
        """
        der_band = np.zeros((self.N*4, 16))
        reg_band = np.zeros((self.N*4, 16))
        for n in range(0, self.N):
            for i, m in enumerate(range(n, min(n+4, self.N))):
                der_band[n*4:n*4+4, i*4:i*4+4] = self.compute_matrix_der_mn(m, n)
                reg_band[n*4:n*4+4, i*4:i*4+4] = self.compute_matrix_reg_mn(m, n)
        return der_band, reg_band

    def compute_sparses_matrices(self, der_band, reg_band):
        """
        :action: Creates sparse banded matrices from bands and stores them in
         class properties
        :param der_band: band for the derivatives part of the normal attitude
         matrix
        :param reg_band: band for the regularization part of the normal attitude
         matrix
        """
        self.attitude_der_matrix = helpers.get_sparse_diagonal_matrix_from_half_band(der_band)
        self.attitude_reg_matrix = helpers.get_sparse_diagonal_matrix_from_half_band(reg_band)

    def compute_matrix_reg_mn(self, m_index, n_index):
        """
        | Ref. Paper [LUDW2011]_ eq. [82]
        | compute :math:`\lambda^2*\\frac{dD_l}{da_m} * \\frac{dD_l}/{da_n^T}`
          (i.e. wrt coeffs)

        :param m_index: [int]
        :param n_index: [int]
        :return: reg_mn [n, m] entry of the regularisation part of the attitude
         normal matrix
        """
        reg_mn = np.zeros((4, 4))
        time_support_spline_m = get_times_in_knot_interval(self.all_obs_times, self.att_knots, m_index, self.M)
        time_support_spline_n = get_times_in_knot_interval(self.all_obs_times, self.att_knots, n_index, self.M)
        time_support_spline_mn = np.sort(helpers.get_lists_intersection(time_support_spline_m, time_support_spline_n))

        for i, t_L in enumerate(time_support_spline_mn):
            left_index = get_left_index(self.att_knots, t=t_L, M=self.M)
            obs_time_index = list(self.all_obs_times).index(t_L)
            # Compute the regulation part
            coeff_basis_sum = compute_coeff_basis_sum(self.att_coeffs, self.att_bases,
                                                      left_index, self.M, obs_time_index)
            dDL_da_n = compute_DL_da_i(coeff_basis_sum, self.att_bases, obs_time_index, n_index)
            dDL_da_m = compute_DL_da_i(coeff_basis_sum, self.att_bases, obs_time_index, m_index)
            reg_mn += self.attitude_regularisation_factor**2 * dDL_da_n @ dDL_da_m.T

        return reg_mn

    def compute_matrix_der_mn(self, m_index, n_index):
        """
        | Ref. Paper [LUDW2011]_ eq. [82]
        | Compute :math:`\\frac{dR_l}{da_n} \\frac{dR_l}{da_m^T}`
          (i.e. wrt coeffs)

        :param m_index: [int]
        :param n_index: [int]
        :return: der_mn, entry [n ,m] of the derivative parts of the attitude
         normal matrix
        """

        der_mn = np.zeros((4, 4))
        time_support_spline_m = get_times_in_knot_interval(self.all_obs_times, self.att_knots, m_index, self.M)
        time_support_spline_n = get_times_in_knot_interval(self.all_obs_times, self.att_knots, n_index, self.M)
        time_support_spline_mn = np.sort(helpers.get_lists_intersection(time_support_spline_m, time_support_spline_n))

        for i, t_L in enumerate(time_support_spline_mn):
            # for i, t_L in enumerate(self.all_obs_times):
            calc_source = self.calc_sources[self.get_source_index(t_L)]
            attitude = self.get_attitude(t_L, unit=False)
            obs_time_index = list(self.all_obs_times).index(t_L)

            # Compute the original objective function part
            dR_dq_AL, dR_dq_AC = compute_dR_dq(calc_source, self.sat, attitude, t_L)
            dR_da_m_AL = dR_da_i(dR_dq_AL, self.att_bases[m_index, obs_time_index])
            dR_da_m_AC = dR_da_i(dR_dq_AC, self.att_bases[m_index, obs_time_index])
            dR_da_n_AL = dR_da_i(dR_dq_AL, self.att_bases[n_index, obs_time_index])
            dR_da_n_AC = dR_da_i(dR_dq_AC, self.att_bases[n_index, obs_time_index])
            der_mn += dR_da_n_AL @ dR_da_m_AL.T + dR_da_n_AC @ dR_da_m_AC.T

        return der_mn

    # ### Implementation with moble's quaternion
    # This functions are not yet ready, but are here to be for a future
    # implementation of numpy-compatible implementation

    def compute_attitude_splines(self):
        s_w = self.attitude_splines[0]
        s_x = self.attitude_splines[1]
        s_y = self.attitude_splines[2]
        s_z = self.attitude_splines[3]
        splines_coeffs = np.array([s_w, s_x, s_y, s_z]).T
        self.discretized_attitude = quaternion.from_float_array(splines_coeffs)
        return self.discretized_attitude

    def get_attitude_from_attitude_array(self, t):
        raise ValueError('This function is not complete')
        pass

    def compute_stuff_for_source(self, s):
        raise ValueError('This function is not complete')
        alpha, delta, parallax, mu_alpha, mu_delta = calc_source.s_params[:]
        params = np.array([alpha, delta, parallax, mu_alpha, mu_delta, calc_source.mu_radial])
        Cu = compute_u(sat, t)  # u in CoMRS frame
        Su = ft.lmn_to_xyz(attitude, Cu)  # u in SRS frame
        phi, zeta = compute_field_angles(Su, double_telescope=False)
        pass


if __name__ == '__main__':
    print('Executing agis.py as main file')
