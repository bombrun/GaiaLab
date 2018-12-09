# -*- coding: utf-8 -*-
"""
File test_gaia_lab.py

File contains some test for GaiaLab main functions

:Author: Luca Zampieri 2018
"""

from source import Source
from satellite import Satellite
from scanner import Scanner
import helpers as helpers
from agis import Calc_source
from agis import Agis
import frame_transformations as ft
import agis_functions as af

from quaternion_implementation import Quaternion
import quaternion


import numpy as np
from scipy import interpolate
from scipy.interpolate import BSpline
from scipy.interpolate import splev
import unittest


class test_Quaternion(unittest.TestCase):

    def setUp(self):
        w, x, y, z = np.random.rand(4)
        self.quat = Quaternion(w, x, y, z)

    def test_conjugate_1(self):
        q2 = self.quat.conjugate().conjugate()
        q1_params = [self.quat.w, self.quat.x, self.quat.y, self.quat.z]
        q2_params = [q2.w, q2.x, q2.y, q2.z]
        for a, b in zip(q1_params, q2_params):
            self.assertEqual(a, b)

    def test_conjugate_2(self):
        self.assertAlmostEqual(self.quat.magnitude,
                               self.quat.conjugate().magnitude)

    def test_unit(self):
        self.assertAlmostEqual(1, self.quat.unit().magnitude)

    def test_inverse(self):
        q = self.quat*self.quat.inverse()
        self.assertAlmostEqual(1, q.w)
        self.assertAlmostEqual(0, q.x)
        self.assertAlmostEqual(0, q.y)
        self.assertAlmostEqual(0, q.z)

    def test_operations(self):
        pass

    def test_rotation_axis_and_angle(self):
        # WARNING: this would fail if a symmetric matrix is generated (unlikely)
        # In that case you should just run again the test
        v1 = np.random.rand(3)
        v2 = np.random.rand(3)
        vector, angle = helpers.get_rotation_vector_and_angle(v1, v2)
        quat = Quaternion(vector=vector, angle=angle)
        vector_bis, angle_bis = quat.rotation_axis_and_angle()
        self.assertAlmostEqual(angle, angle_bis)
        np.testing.assert_array_almost_equal(vector_bis, vector)


class test_source(unittest.TestCase):

    def setUp(self):
        self.source = Source('test', 0, 1, 2, 3, 4, 5)

    def test_init_param_types(self):
        pass


class test_satellite(unittest.TestCase):

    def setUp(self):
        t_init = 0
        t_end = 10
        my_dt = 1/24
        self.sat = Satellite(t_init, t_end, my_dt)

    def test_init_state(self):
        self.assertRaises(TypeError, self.sat.attitude, Satellite)


class test_scanner(unittest.TestCase):
    def setUp(self):
        self.scanner = Scanner()

    def test_scanner_by_day(self):
        pass


class test_agis_functions(unittest.TestCase):

    def test_compute_du_dparallax(self):
        b_G = [1, 2, 3]
        r = np.random.rand(3)
        self.assertRaises(TypeError, af.compute_du_dparallax, r, b_G)
        b_G = np.random.rand(3)
        self.assertRaises(ValueError, af.compute_du_dparallax, r, b_G)
        r.shape = (3, 1)
        result = af.compute_du_dparallax(r, b_G)
        b1, b2, b3 = b_G[:]
        r1, r2, r3 = r.flatten()[0], r.flatten()[1], r.flatten()[2]
        desired_result = -np.array([(1-r1**2)*b1 + (-r1*r2)*b2 + (-r1*r3)*b3,
                                   (-r2*r1)*b1 + (1-r2**2)*b2 + (-r2*r3)*b3,
                                   (-r3*r1)*b1 + (-r3*r2)*b2 + (1-r3**2)*b3])
        desired_result.shape = 3
        np.testing.assert_equal(result, desired_result)

    def test_compute_field_angles(self):
        Su = [1, 0, 0, 0]
        self.assertRaises(TypeError, af.compute_field_angles, Su)
        Su = np.array(Su)
        self.assertRaises(ValueError, af.compute_field_angles, Su)
        Su_list = [np.array([1, 0, 0]),
                   np.array([0, 1, 0]),
                   np.array([0, 0, 1])]
        etas_res = [0, np.pi/2, 0]
        zetas_res = [0, 0, np.pi/2]
        for Su, eta_res, zeta_res in zip(Su_list, etas_res, zetas_res):
            eta, zeta = af.compute_field_angles(Su)
            self.assertEqual(eta, eta_res)
            self.assertEqual(zeta, zeta_res)

    def test_extend_knots(self):
        k = 4
        knots = [0, 1, 2, 3, 4]
        extended_knots = af.extend_knots(knots, k)
        self.assertEqual(len(extended_knots), len(knots)+2*k)

    def test_extract_coeffs_knots_from_splines(self):
        k = 4
        length = 5
        x = np.arange(length)
        y = np.random.rand(length)
        spline = interpolate.InterpolatedUnivariateSpline(x, y, k=k)
        spline_list = [spline]

        coeffs, knots, splines = af.extract_coeffs_knots_from_splines([spline], k)
        self.assertEqual(len(coeffs), len(spline_list))

    def test_get_basis_Bsplines(self):
        k = 4  # spline order (degree+1)
        length = 100
        x = np.arange(length)
        y = np.random.rand(length)
        x_eval = x
        spline = interpolate.InterpolatedUnivariateSpline(x, y, k=k)
        spline_list = [spline]

        coeffs, knots, splines = af.extract_coeffs_knots_from_splines([spline], k)
        coeffs, knots = coeffs.flatten(), knots.flatten()
        bases = af.get_basis_Bsplines(knots, coeffs, k, knots)
        for i in range(k, bases.shape[0] - k):
            non_zero = np.where(bases[i, :] != 0)[0]
            self.assertEqual(non_zero.shape[0], k)
            # once we checked that they are k non-zero we can check their position with a sum:
            self.assertEqual(non_zero.sum(), (i+1)*k+(0+k-1)*k/2)  # arithmetic sum

    def test_get_times_in_knot_interval(self):
        """ Test if the time interval is consistent"""
        M = 4
        index = 55
        my_min, my_max = (0, 100)
        time_array = np.linspace(my_min, my_max, num=100)
        knots = np.linspace(my_min, my_max, num=100)
        times = af.get_times_in_knot_interval(time_array, knots, index, M)
        self.assertTrue(knots[index] < times[0])
        self.assertTrue(times[-1] < knots[index+M])


class test_agis_2(unittest.TestCase):

    def setUp(self):
        num_observations = 1
        t_init = 0
        t_end = 10
        my_dt = 1/24
        sat = Satellite(t_init, t_end, my_dt)
        t_list = np.linspace(t_init, t_end, num_observations)
        source = Source('test', 0, 1, 2, 3, 4, 5)

        source.reset()
        s = np.zeros(5)
        s[0] = source.alpha / 2
        s[1] = source.delta / 2
        s[2] = source.parallax / 2
        s[3] = source.mu_alpha_dx
        s[4] = source.mu_delta

        calc_source = Calc_source('calc_test', t_list, s, source.mu_radial)
        self.solver = Agis(sat, [calc_source], [source])

    # def test_error_function(self):
        # pass
        # self.assertTrue(0 <= self.solver.error_function())


class test_agis(unittest.TestCase):

    def setUp(self):
        t_init = 0  # 1/24/60
        t_end = t_init + 1/24/60  # 365*5
        my_dt = 1/24/60/10  # [days]
        spline_degree = 3
        gaia = Satellite(ti=t_init, tf=t_end, dt=my_dt, k=spline_degree)
        self.gaia = gaia
        my_times = np.linspace(t_init, t_end, num=100, endpoint=False)
        real_sources = []
        calc_sources = []
        for t in my_times:
            alpha, delta = af.generate_observation_wrt_attitude(gaia.func_attitude(t))
            real_src_tmp = Source(str(t), np.degrees(alpha), np.degrees(delta), 0, 0, 0, 0)
            calc_src_tmp = Calc_source('calc_'+str(t), [t], real_src_tmp.get_parameters()[0:5],
                                       real_src_tmp.get_parameters()[5])
            real_sources.append(real_src_tmp)
            calc_sources.append(calc_src_tmp)
        # test if source and calc source are equal (as they should be)
        np.testing.assert_array_almost_equal(np.array(real_sources[0].get_parameters()[0:5]), calc_sources[0].s_params)
        # create Solver
        self.Solver = Agis(gaia, calc_sources, real_sources, attitude_splines=[gaia.s_w, gaia.s_x, gaia.s_y, gaia.s_z],
                           spline_degree=spline_degree, attitude_regularisation_factor=1e-3)

    def test_unicity_of_knots(self):
        """[Attitude] test if knots are the same for each component"""
        gaia = self.gaia
        internal_knots = self.Solver.att_knots[self.Solver.k:-self.Solver.k]
        for gaia_knots in [gaia.s_w.get_knots(), gaia.s_x.get_knots(), gaia.s_y.get_knots(), gaia.s_z.get_knots()]:
            np.testing.assert_array_almost_equal(internal_knots, gaia_knots)

    def test_compute_coeff_basis_sum(self):
        """ [Attitude] Tests some ways of forming a spline """
        # given the spline:
        # m = 10  # [0-100]  # spline number
        m = np.random.randint(low=0, high=self.Solver.N)
        M = self.Solver.M
        knots = self.Solver.att_knots
        coeffs = self.Solver.att_coeffs[0]
        bases = self.Solver.att_bases
        observed_times = self.Solver.all_obs_times[(knots[m] <= self.Solver.all_obs_times) &
                                                   (self.Solver.all_obs_times <= knots[m+M])]
        if not list(observed_times):
            raise ValueError('not observed times in interval')
        t = observed_times[5]
        index = np.where(self.Solver.all_obs_times == t)[0][0]

        L = af.get_left_index(self.Solver.att_knots, t, M)
        b_list = []
        for i, n in enumerate(range(L-M+1, L+1)):  # last +1 because range does not inlude the last point
            coeff = coeffs[n]
            bspline = bases[n]
            b_list.append(coeff*bspline)
        my_spline = af.compute_coeff_basis_sum(self.Solver.att_coeffs, self.Solver.att_bases, L, M, index)
        ref_spline1 = BSpline(self.Solver.att_knots, self.Solver.att_coeffs[0], k=self.Solver.k)(t)
        ref_spline2 = sum([coef*bspline for coef, bspline in zip(self.Solver.att_coeffs[0], self.Solver.att_bases)])
        ref_spline3 = sum(b_list)
        ref_spline4 = np.sum(self.Solver.att_bases[L-M:L+1, index] * self.Solver.att_coeffs[:, L-M:L+1], axis=1)

        self.assertEqual(my_spline[0], ref_spline1)
        self.assertEqual(my_spline[0], ref_spline2[index])
        self.assertEqual(my_spline[0], ref_spline3[index])
        # self.assertEqual(my_spline[0], ref_spline4[0])

    def test_initialisation_attitude(self):
        """ [Attitude] Test if generated source comply with the copied (from satellite) attitude"""
        self.Solver.actualise_splines()
        error = self.Solver.error_function()
        self.assertAlmostEqual(error, 0, delta=1e-25)

    def test_dDL_da_i(self):
        """ [attitude] test consistency of derivative of the attitude deviation from unity"""
        n_index, m_index = (4, 5)
        M = self.Solver.M
        knots = self.Solver.att_knots
        coeffs = self.Solver.att_coeffs
        bases = self.Solver.att_bases

        # Get the times
        observed_times_m = af.get_times_in_knot_interval(self.Solver.all_obs_times, knots, m_index, M)
        observed_times_n = af.get_times_in_knot_interval(self.Solver.all_obs_times, knots, n_index, M)
        observed_times_mn = helpers.get_lists_intersection(observed_times_m, observed_times_n)
        t_L = observed_times_mn[0]

        obs_time_index = list(self.Solver.all_obs_times).index(t_L)
        L = af.get_left_index(knots, t_L, M=M)
        attitude = self.Solver.get_attitude(t_L)

        coeff_basis_sum = af.compute_coeff_basis_sum(coeffs, bases, L, M, obs_time_index)
        dDL_da_1 = af.compute_DL_da_i(coeff_basis_sum, bases,
                                      obs_time_index, n_index)
        dDL_da_2 = af.compute_DL_da_i_from_attitude(attitude, bases,
                                                    obs_time_index, n_index)
        self.assertAlmostEqual(dDL_da_1[0], dDL_da_2[0], delta=1e-15)
        self.assertAlmostEqual(dDL_da_1[2], dDL_da_2[2], delta=1e-15)
        np.testing.assert_array_almost_equal(dDL_da_1, dDL_da_2, decimal=15)

    def test_equivalence_with_sparse_attitude(self):
        """
        Tests if attitude update matrix is the same if computed with the sparse
        version or the full version. """
        der_band, reg_band = self.Solver.compute_attitude_banded_derivative_and_regularisation_matrices()
        self.Solver.compute_sparses_matrices(der_band, reg_band)
        sparse_matrix = self.Solver.attitude_der_matrix + self.Solver.attitude_reg_matrix
        full_matrix = self.Solver.compute_attitude_LHS()
        np.testing.assert_array_almost_equal(sparse_matrix.toarray(), full_matrix)


class test_helpers(unittest.TestCase):

    def test_compute_angle(self):
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        self.assertAlmostEqual(0, helpers.compute_angle(v1, v1))
        self.assertAlmostEqual(np.radians(90), helpers.compute_angle(v1, v2))

    def test_get_rotation_matrix(self):
        v1 = np.random.rand(3)
        v2 = np.random.rand(3)
        v1 = v1/np.linalg.norm(v1)
        v2 = v2/np.linalg.norm(v2)
        rot = helpers.get_rotation_matrix(v1, v2)
        v2_bis = rot@v1.T
        for i in range(3):
            self.assertAlmostEqual(v2[i], v2_bis[i])

    def test_get_rotation_vector_and_angle(self):
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        v3 = np.array([0, 0, 1])
        vector, angle = helpers.get_rotation_vector_and_angle(v1, v2)
        for i in range(vector.shape[0]):
            self.assertAlmostEqual(v3[i], vector[i])
        self.assertAlmostEqual(np.pi / 2, angle)

    def test_get_lists_intersection(self):
        """ test if it returns indeed the intersection of the lists"""
        list1 = [0, 1, 2, 3, 4, 5]
        list2 = [3, 4, 5, 6, 7, 8]
        intersection = helpers.get_lists_intersection(list1, list2)
        self.assertEqual(len(intersection), 3)
        self.assertEqual(intersection[0], 3)
        self.assertEqual(intersection[1], 4)
        self.assertEqual(intersection[2], 5)


class test_frame_transformations(unittest.TestCase):

    def setUp(self):
        pass

    def test_rotation_with_quat(self):
        v1 = np.random.rand(3)
        v2 = np.random.rand(3)
        v1 = v1/np.linalg.norm(v1)
        v2 = v2/np.linalg.norm(v2)
        vector, angle = helpers.get_rotation_vector_and_angle(v1, v2)
        quat = quaternion.from_rotation_vector(vector*angle)
        v2_bis = ft.rotate_by_quaternion(quat, v1)
        for i in range(vector.shape[0]):
            self.assertAlmostEqual(v2[i], v2_bis[i])

    def test_rotation_against_quat(self):
        """ Test that rotating with quaternion or matrix is equivalent"""
        v1 = np.random.rand(3)
        v2 = np.random.rand(3)
        v1 = v1/np.linalg.norm(v1)
        v2 = v2/np.linalg.norm(v2)

        rot = helpers.get_rotation_matrix(v1, v2)
        vector, angle = helpers.get_rotation_vector_and_angle(v1, v2)
        quat = Quaternion(vector=vector, angle=angle).unit()
        rot_quat = quat.basis()
        for x_row, y_row in zip(rot, rot_quat):
            for a, b in zip(x_row, y_row):
                self.assertAlmostEqual(a, b)
        # v2_bis = rot@v1.T
        # v2_tris = rot_quat@v1.T


if __name__ == '__main__':
    unittest.main(verbosity=3)
