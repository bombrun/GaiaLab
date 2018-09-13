
import unittest
from . import gaia_geometric_toymodel as ggs
from .quaternion import Quaternion
import numpy as np

class test_sky(unittest.TestCase):

    def setUp(self):
        self.n = 5
        self.sky = ggs.Sky(self.n)
        self.assertEqual(len(self.sky.elements), self.n)
        self.assertEqual(type(self.sky.elements[0].coor), np.ndarray)
        for i in self.sky.elements:
            self.assertAlmostEqual(np.linalg.norm(i.coor), 1)

    def test_types(self):
        self.assertRaises(TypeError, self.n, float)
        self.assertRaises(TypeError, self.n, True)
        self.assertRaises(TypeError, self.n, 'string')

class test_source(unittest.TestCase):

    def setUp(self):
        self.alpha = np.random.uniform(0, 7)
        self.delta = np.random.uniform(0, 7)
        self.source = ggs.Source(self.alpha, self.delta)
        self.assertAlmostEqual(np.linalg.norm(self.source.coor), 1)

class test_satellite(unittest.TestCase):

    def setUp(self):
        self.n = 5
        self.satellite = ggs.Satellite(self.n)

    def test_types(self):
        self.assertRaises(TypeError, self.satellite.S, True)
        self.assertRaises(TypeError, self.satellite.S, 'string')
        self.assertRaises(TypeError, self.satellite.S, 3+2j)

        self.assertRaises(TypeError, self.satellite.epsilon, True)
        self.assertRaises(TypeError, self.satellite.epsilon, 'string')
        self.assertRaises(TypeError, self.satellite.epsilon, 3 + 2j)

        self.assertRaises(TypeError, self.satellite.xi, True)
        self.assertRaises(TypeError, self.satellite.xi, 'string')
        self.assertRaises(TypeError, self.satellite.xi, 3 + 2j)

        self.assertRaises(TypeError, self.satellite.wz, True)
        self.assertRaises(TypeError, self.satellite.wz, 'string')
        self.assertRaises(TypeError, self.satellite.wz, 3 + 2j)

class test_attitude(unittest.TestCase):
    
    def setUp(self):
        self.att = ggs.Attitude()
        if isinstance(self.att, ggs.Satellite) != True:
            raise TypeError('Attitude is not a Satellite object')
    def test_init_state(self):
        self.assertEqual(self.att.nu, 0)
        self.assertEqual(self.att._lambda, 0)
        self.assertEqual(self.att.omega, 0)
        self.assertEqual(self.att._beta, 0)
        self.assertEqual(self.att.t, 0)

        if isinstance(self.att.init_attitude(), Quaternion) is False:
            raise TypeError('Init Attitude not a quaternion object')

    def test_reset(self):
        self.att.update(np.random.uniform(0,10))
        self.att.reset()
        self.assertEqual(self.att.nu, 0)
        self.assertEqual(self.att._lambda, 0)
        self.assertEqual(self.att.omega, 0)
        self.assertEqual(self.att._beta, 0)
        self.assertEqual(self.att.t, 0)
    
        if isinstance(self.att.z, np.ndarray) == False:
            raise TypeError('z is not a vector')
    
    def test_update(self):
        dt = np.random.uniform(0,1)
        self.att.update(dt)

        if isinstance(self.att.attitude, Quaternion) == False:
            raise TypeError('updated satellite.attitude is not a quaternion')

    def test_long_reset_to_time(self):
        t = np.random.uniform(0,10)
        dt = np.random.uniform(0, 1)
        self.att.long_reset_to_time(t, dt)

class test_scanner(unittest.TestCase):
    
    def setUp(self):
        ccd = np.random.uniform(0, 1)
        delta_z = np.random.uniform(0, 0.5)
        delta_y = np.random.uniform(0, 0.3)
        self.scan = ggs.Scanner(ccd, delta_z, delta_y)

    def test_types(self):
        self.assertRaises(TypeError, self.scan.delta_z, True)
        self.assertRaises(TypeError, self.scan.delta_z, 'string')
        self.assertRaises(TypeError, self.scan.delta_y, True)
        self.assertRaises(TypeError, self.scan.delta_y, 'string')
        self.assertRaises(TypeError, self.scan.ccd, True)
        self.assertRaises(TypeError, self.scan.ccd, 'string')
    def test_reset(self):
        self.scan.reset_memory()
        self.assertEqual(len(self.scan.obs_times), 0)
        self.assertEqual(len(self.scan.telescope_positions), 0)
        self.assertEqual(len(self.scan.times_deep_scan), 0)

if __name__ == '__main__':
    unittest.main()




    
