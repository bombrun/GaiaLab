
import unittest 
from NSL import*

class SatelliteTest(unittest.TestCase):
    
    def setUp(self):
        self.satellite = Satellite()
        
    def test_InitAttitude(self):
        if isinstance(self.satellite.InitAttitude(), Quaternion) == False:
            raise Exception('Satellite attitude not a quaternion object')
        
    def test_Reset(self):
        self.satellite.Update(np.random.uniform(0,10))
        self.satellite.Reset()
        
        self.assertEqual(self.satellite.nu0, self.satellite.nu)
        self.assertEqual(self.satellite.omega0, self.satellite.omega)
        self.assertEqual(self.satellite.l0, self.satellite.l)
        self.assertEqual(self.satellite.beta0, self.satellite.beta)
        self.assertEqual(self.satellite.t, 0)
    
        if isinstance(self.satellite.z_, np.ndarray) == False:
            raise Exception('z is not a vector')
    
    def test_Update(self):
        dt = np.random.uniform(0,10)
        self.satellite.Update(dt)
        
        self.assertEqual(self.satellite.nudot, self.satellite.dNu/dt)
        self.assertEqual(self.satellite.ldot, self.satellite.dL/dt)
                
        if isinstance(self.satellite.attitude, Quaternion) == False:
            raise Exception('updated satellite.attitude is not a quaternion')
            
    def test_GetAttitude(self):
        dt = np.random.uniform(0,1)
        t = np.random.uniform(0,10)
        #need to check that it updates the attitude i times. 
        
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
        
    def test_GetFV1(self):
        if isinstance(self.attitude, Quaternion) == False:
            raise ValueError('self.scanner.GetFV1 takes only quaternion as attitude')
        if isinstance(self.scanner.GetFV1(self.attitude), np.dnarray) == False:
            raise Exception('scanner.GetFV1 does not return np.array')
        self.assertEqual(len(self.scanner.GetFV1(self.attitude)),3)
        
    def test_GetFV2(self):
        if isinstance(self.attitude, Quaternion) == False:
            raise ValueError('self.scanner.GetFV2 takes only quaternion as attitude')
        if isinstance(self.scanner.GetFV2(self.attitude), np.dnarray) == False:
            raise Exception('scanner.GetFV2 does not return np.array')
        self.assertEqual(len(self.scanner.GetFV2(self.attitude)),3)
        
    def Inst_Scan_FV1(self):
        findings_1 = self.scanner.Inst_Scan_FV1(self.sky, self.attitude)
        for i in findings_1:
            if isinstance(i, np.dnarray) == False:
                return Exception('element in found stars in not a np.array')
      
if __name__ == "__main__":
    unittest.main




    