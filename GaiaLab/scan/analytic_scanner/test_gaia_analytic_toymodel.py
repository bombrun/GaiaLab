
import unittest 
from . import gaia_analytic_toymodel as gat
import numpy as np

class test_source(unittest.TestCase):

    def setUp(self):
        self.source = gat.Source('test',0,1,2,3,4,5)

    def test_init_param_types(self):
        self.assertRaises(TypeError, self.source.parallax, 3 +5j)
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
        self.satellite = gat.Satellite

class test_attitude(unittest.TestCase):

    def setUp(self):
        self.att = gat.Attitude()

    def test_init_state(self):
        self.assertRaises(TypeError, self.att.attitude, gat.Attitude)

class test_scanner(unittest.TestCase):
    def setUp(self):
        self.scan = gat.Scanner()

if __name__ == '__main__':
    unittest.main()




    
