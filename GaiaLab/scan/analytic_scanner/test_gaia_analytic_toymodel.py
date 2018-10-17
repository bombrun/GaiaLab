
import unittest
from source import Source
from satellite import Satellite
from scanner import Scanner
import helpers as helpers
import frame_transformations as ft

import numpy as np


class test_source(unittest.TestCase):

    def setUp(self):
        self.source = Source('test', 0, 1, 2, 3, 4, 5)

    def test_init_param_types(self):
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
        self.sat = Satellite()

    def test_init_state(self):
        self.assertRaises(TypeError, self.sat.attitude, Satellite)


class test_scanner(unittest.TestCase):
    def setUp(self):
        self.scan = Scanner()


class test_agis(unittest.TestCase):

    def setUp(self):
        pass

    def test_init_state(self):
        pass


class test_helpers(unittest.TestCase):

    def test_compute_angle(self):
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        self.assertAlmostEqual(0, helpers.compute_angle(v1, v1))
        self.assertAlmostEqual(np.radians(90), helpers.compute_angle(v1, v2))


class test_frame_transformations(unittest.TestCase):

    def test_get_rotation_matrix(self):
        v1 = np.random.rand(3)
        v2 = np.random.rand(3)
        rot = ft.get_rotation_matrix(v1, v2)
        v2_bis = rot@v1.T
        for i in range(3):
            # self.assertAlmostEqual(v2[i], v2_bis[i])
            pass


if __name__ == '__main__':
    unittest.main(verbosity=3)
