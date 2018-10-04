
import unittest
from source import Source
from satellite import Satellite
from scanner import Scanner

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


if __name__ == '__main__':
    unittest.main()
