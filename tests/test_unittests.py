import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from gaialab.scanner.source import Source
from gaialab.scanner.satellite import Satellite
from gaialab.scanner.scanner import Scanner
import gaialab.scanner.helpers as helpers
from gaialab.scanner.agis import Calc_source
from gaialab.scanner.agis import Agis
import gaialab.scanner.frame_transformations as ft
import gaialab.scanner.agis_functions as af
import gaialab.scanner.constants as const

import numpy as np
import quaternion

from scipy import interpolate
from scipy.interpolate import BSpline
from scipy.interpolate import splev
import unittest
from gaialab.solver.newsource import compute_du_ds
from gaialab.scanner.source import compute_topocentric_direction

p=np.array([[0],
            [0],
            [0]])
q=np.array([[0],
            [0],
            [0]])
r=np.array([[0],
            [0],
            [0]])
gaia = Satellite(0, 365*5, 1/24)
t_l=0
q_l=np.quaternion(1,1,1,np.pi)
print ("q_l=", q_l)
b_G = gaia.ephemeris_bcrs(t_l)
tau = t_l - const.t_ep
x=af.compute_du_dparallax(r, b_G)
print ("x=", x)
du_ds_CoMRS = [p, q, af.compute_du_dparallax(r, b_G), p*tau, q*tau]
print (du_ds_CoMRS)
A=compute_du_ds(p,q,r,q_l,t_l)
print ("A=",A)

astro_parameters=[101.28, -16.7161, 379.21, -546.05, -1223.14, -7.6]
t_init = 0
t_end =  365*5
my_dt = 1/24 # [days]
gaia = Satellite(ti=t_init, tf=t_end, dt= my_dt)
B=compute_topocentric_direction(astro_parameters, gaia, t_init)
print ("B=", B)


#GOOD
class test_du_ds(unittest.TestCase):
    def test_dimension (self):
        gaia = Satellite(0, 365*5, 1/24)
        A= compute_du_ds(np.random.rand(3,1),np.random.rand(3,1),np.random.rand(3,1),gaia.func_attitude(0),0)
        self.assertEqual(A.shape, (5,3))

if __name__ == '__main__':
    unittest.main(verbosity=3)
