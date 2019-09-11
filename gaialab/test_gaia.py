import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from source import Source
from satellite import Satellite
from scanner import Scanner
import helpers as helpers
from source import Calc_source
from agis import Agis
import frame_transformations as ft
import agis_functions as af
import constants as const

import numpy as np
import quaternion


import unittest
from source import compute_u
from source import Source


#p=np.array([[0],
#            [0],
#            [0]])
#q=np.array([[0],
#            [0],
#            [0]])
#r=np.array([[0],
#            [0],
#            [0]])
#gaia = Satellite(0, 365*5, 1/24)
#t_l=0
#q_l=np.quaternion(1,1,1,np.pi)
#b_G = gaia.ephemeris_bcrs(t_l)
#tau = t_l - const.t_ep
#x=af.compute_du_dparallax(r, b_G)
#du_ds_CoMRS = [p, q, af.compute_du_dparallax(r, b_G), p*tau, q*tau]
#A=compute_du_ds(p,q,r,q_l,t_l)

param=[101.28, -16.7161, 379.21, -546.05, -1223.14, 0]
t_init = 0
t_end =  365*5
my_dt = 1/24 # [days]
gaia = Satellite(ti=t_init, tf=t_end, dt= my_dt)
sirio = Source("sirio", 101.28, -16.7161, 379.21, -546.05, -1223.14, 0 )
B=sirio.compute_u(param, gaia, t_init)
print ("B=", B)

def unit_topocentric_function( satellite, t):
    """
    Compute the topocentric_function direction

    :param satellite: satellite [class object]
    :return: [array] (x,y,z) direction-vector of the star from the satellite's lmn frame.
    """
    # self.set_time(0)  # (float(t))
    param = [101.28, -16.7161, 379.21, -546.05, -1223.14, -7.6]
    return compute_u(param, satellite, t)


A=unit_topocentric_function(satellite=gaia, t=0)
print ("A=",A)


#class test_compute_u(unittest.TestCase):
#    def test_same_function (self):
#        gaia = Satellite(0, 365*5, 1/24)
#        param=[101.28, -16.7161, 379.21, -546.05, -1223.14, -7.6]
#        A= compute_u(param, gaia, 0)
#        B= unit_topocentric_function(satellite=gaia, t=0)
#        self.assertEqual(A, B)
#
#if __name__ == '__main__':
#    unittest.main(verbosity=3)


alpha= 101.28
delta=-16.7161
p, q, r = ft.compute_pqr(alpha, delta)
print("r=",r)
rt=r.transpose()
print ("rt=",rt)
print (rt.shape)
