from source import Source
from satellite import Satellite
from scanner import Scanner
import helpers as helpers
import frame_transformations as ft
import solver as solver

import quaternion

import numpy as np
from scipy import interpolate
from scipy.interpolate import BSpline
from scipy.interpolate import splev
import unittest

class test_source(unittest.TestCase):

    def setUp(self):
        self.source = Source('test', 0, 1, 2, 3, 4, 5, 6, 7)

    def test_init_param_types(self):
        #[p1, p2, p3, p4, p5, p6, p7, p8]=self.source.get_parameters(t=0)
        #self.assertEqual(type(p1), 'numpy.float64')
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
        quat = np.quaternion(vector=vector, angle=angle).unit()
        rot_quat = quat.basis()
        for x_row, y_row in zip(rot, rot_quat):
            for a, b in zip(x_row, y_row):
                self.assertAlmostEqual(a, b)
        # v2_bis = rot@v1.T
        # v2_tris = rot_quat@v1.T


if __name__ == '__main__':
    unittest.main(verbosity=3)
