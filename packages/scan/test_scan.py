
import unittest 
from NSL import*
import numpy as np

class SatelliteTest(unittest.TestCase):
    
    def setUp(self):
        self.satellite = Satellite()
        
    def test_init_attitude(self):
        if isinstance(self.satellite.init_attitude(), Quaternion) == False:
            raise Exception('Satellite attitude not a quaternion object')
        
    def test_reset(self):
        self.satellite.update(np.random.uniform(0,10))
        self.satellite.reset()
        
        self.assertEqual(self.satellite.nu, self.satellite.nu)
        self.assertEqual(self.satellite.omega, self.satellite.omega)
        self.assertEqual(self.satellite.l, self.satellite.l)
        self.assertEqual(self.satellite.beta, self.satellite.beta)
        self.assertEqual(self.satellite.t, 0)
    
        if isinstance(self.satellite.z_, np.ndarray) == False:
            raise Exception('z is not a vector')
    
    def test_update(self):
        dt = np.random.uniform(0,10)
        self.satellite.update(dt)
        
        #self.assertEqual(self.satellite.nudot, self.satellite.dNu/dt)
        #self.assertEqual(self.satellite.ldot, self.satellite.dL/dt)
                
        if isinstance(self.satellite.attitude, Quaternion) == False:
            raise Exception('updated satellite.attitude is not a quaternion')
           
    def test_move(self):
        ti = np.random.uniform(0,10)
        tf = np.random.uniform(0,10)
        dt = np.random.uniform(0,1)
        
        self.satellite.move(ti, tf, dt)
        
        for obj in self.satellite.storing_list:
            self.assertEqual(len(obj), 9)

    def test_reset_to_time(self):
        t = np.random.uniform(0,10)
        self.satellite.reset_to_time(t)
        
        if len(self.storing_list) != 0:
            self.assertAlmostEqual(self.satellite.t, t)
        else:
            self.assertEqual(self.satellite.t, 0.0)
            
    def test_GetXAxis(self):
        #need to check that x_quat is quaternion and then that x_ is a vector.
        t = np.random.uniform(0,10)
        x_ = self.satellite.GetXAxis(t)
        
        if isinstance(x_, np.ndarray) == False:
            raise Exception('x_ is not a vector')
        
class SkyTest(unittest.TestCase):
    
    def setUp(self):
        self.n = 5
        self.sky = Sky(self.n)
               
    def test_initSky(self):
        self.assertEqual(len(self.sky.elements), self.n)
        self.assertEqual(self.sky.elements[0].coor, np.dnarray)

class Source(unittest.TestCase):
    
        def setUp(self):
            self.alpha = np.random.uniform(0,7)
            self.delta = np.random.uniform(0,7)
            self.source = Source(self.alpha, self.delta)
            
        #def test_initSource(self): 
        
class ScannerTest(unittest.TestCase):
    
    def setUp(self):
        self.tolerance = np.random.uniform(0,0.5)
        self.scanner = Scanner(self.tolerance)
        self.attitude = Quaternion(np.random.uniform(-1,1), np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1))
        self.sky = Sky(5)
        
    def test_intercept(self):
        self.star = Source()
        self.satellite = Satellite()

        if len(self.satellite.storing_list) = 0:
            raise ImportError('satellite.storing_list empty')

if __name__ == "__main__":
    unittest.main




    