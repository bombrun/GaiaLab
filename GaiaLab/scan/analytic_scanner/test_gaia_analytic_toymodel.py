
import unittest
from source import Source
from satellite import Satellite
from scanner import Scanner
import helpers as helpers
from agis import Calc_source
from agis import Agis
import frame_transformations as ft
from quaternion import Quaternion
import agis_functions as af

import numpy as np


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


class test_source(unittest.TestCase):

    def setUp(self):
        self.source = Source('test', 0, 1, 2, 3, 4, 5)

    def test_init_param_types(self):
        self.assertRaises(TypeError, self.source.alpha, 3 + 5j)
        self.assertRaises(TypeError, self.source.alpha, True)
        self.assertRaises(TypeError, self.source.alpha, 'string')

        self.assertRaises(TypeError, self.source.parallax, 3 + 5j)
        self.assertRaises(TypeError, self.source.parallax, True)
        self.assertRaises(TypeError, self.source.parallax, 'string')

        self.assertRaises(TypeError, self.source.mu_alpha_dx, 3 + 5j)
        self.assertRaises(TypeError, self.source.mu_alpha_dx, True)
        self.assertRaises(TypeError, self.source.mu_alpha_dx, 'string')

        self.assertRaises(TypeError, self.source.mu_delta, 3 + 5j)
        self.assertRaises(TypeError, self.source.mu_delta, True)
        self.assertRaises(TypeError, self.source.mu_delta, 'string')


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
        self.scan = Scanner()


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


class test_agis(unittest.TestCase):

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

    def test_error_function(self):
        self.assertTrue(0 <= self.solver.error_function())


class test_helpers(unittest.TestCase):

    def test_compute_angle(self):
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        self.assertAlmostEqual(0, helpers.compute_angle(v1, v1))
        self.assertAlmostEqual(np.radians(90), helpers.compute_angle(v1, v2))


class test_frame_transformations(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_rotation_matrix(self):
        v1 = np.random.rand(3)
        v2 = np.random.rand(3)
        v1 = v1/np.linalg.norm(v1)
        v2 = v2/np.linalg.norm(v2)
        rot = ft.get_rotation_matrix(v1, v2)
        v2_bis = rot@v1.T
        for i in range(3):
            self.assertAlmostEqual(v2[i], v2_bis[i])

    def test_get_rotation_vector_and_angle(self):
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        v3 = np.array([0, 0, 1])
        vector, angle = ft.get_rotation_vector_and_angle(v1, v2)
        for i in range(vector.shape[0]):
            self.assertAlmostEqual(v3[i], vector[i])
        self.assertAlmostEqual(np.pi / 2, angle)

    def test_rotation_with_quat(self):
        v1 = np.random.rand(3)
        v2 = np.random.rand(3)
        v1 = v1/np.linalg.norm(v1)
        v2 = v2/np.linalg.norm(v2)
        vector, angle = ft.get_rotation_vector_and_angle(v1, v2)
        quat = ft.rotation_to_quat(vector, angle)
        v2_bis = ft.rotate_by_quaternion(quat, v1)
        for i in range(vector.shape[0]):
            self.assertAlmostEqual(v2[i], v2_bis[i])

    def test_rotation_against_quat(self):
        """ Test that rotating with quaternion or matrix is equivalent"""
        v1 = np.random.rand(3)
        v2 = np.random.rand(3)
        v1 = v1/np.linalg.norm(v1)
        v2 = v2/np.linalg.norm(v2)

        rot = ft.get_rotation_matrix(v1, v2)
        vector, angle = ft.get_rotation_vector_and_angle(v1, v2)
        quat = ft.rotation_to_quat(vector, angle).unit()
        rot_quat = quat.basis()
        for x_row, y_row in zip(rot, rot_quat):
            for a, b in zip(x_row, y_row):
                self.assertAlmostEqual(a, b)
        # v2_bis = rot@v1.T
        # v2_tris = rot_quat@v1.T


if __name__ == '__main__':
    unittest.main(verbosity=3)
