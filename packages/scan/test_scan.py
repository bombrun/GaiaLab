
import unittest 
from NSL import *
import numpy as np

class AttitudeTest(unittest.TestCase):
    
    def setUp(self):
        self.att = Attitude()

        
    def test_init_state(self):
        self.assertEqual(self.att.nu, 0)
        self.assertEqual(self.att._lambda, 0)
        self.assertEqual(self.att.omega, 0)
        self.assertEqual(self.att._beta, 0)
        self.assertEqual(self.att.t, 0)

        if isinstance(self.att.init_attitude(), Quaternion) is False:
            raise Exception('Init Attitude not a quaternion object')


    def test_reset(self):
        self.att.update(np.random.uniform(0,10))
        self.att.reset()
        
        self.assertEqual(self.att.nu, 0)
        self.assertEqual(self.att._lambda, 0)
        self.assertEqual(self.att.omega, 0)
        self.assertEqual(self.att._beta, 0)
        self.assertEqual(self.att.t, 0)
    
        if isinstance(self.att.z, np.ndarray) == False:
            raise Exception('z is not a vector')
    
    def test_update(self):
        dt = np.random.uniform(0,1)
        self.att.update(dt)

        if isinstance(self.att.attitude, Quaternion) == False:
            raise Exception('updated satellite.attitude is not a quaternion')

    def test_long_reset_to_time(self):
        t = np.random.uniform(0,10)
        dt = np.random.uniform(0, 1)
        self.att.long_reset_to_time(t, dt)


class SkyTest(unittest.TestCase):
    
    def setUp(self):
        self.n = 5
        self.sky = Sky(self.n)
               
    def test_initSky(self):
        self.assertEqual(len(self.sky.elements), self.n)
        self.assertEqual(type(self.sky.elements[0].coor), np.ndarray)

class Source(unittest.TestCase):
    
        def setUp(self):
            self.alpha = np.random.uniform(0,7)
            self.delta = np.random.uniform(0,7)
            self.source = Source(self.alpha, self.delta)

class ScannerTest(unittest.TestCase):
    
    def setUp(self):
        ccd = np.random.uniform(0, 1)
        delta_z = np.random.uniform(0, 0.5)
        delta_y = np.random.uniform(0, 0.3)
        self.scan = Scanner(ccd, delta_z, delta_y)


if __name__ == '__main__':
    unittest.main()




    