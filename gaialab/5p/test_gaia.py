import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


#from source import Source
from satellite import Satellite
from scanner import Scanner
import helpers as helpers
from source import Calc_source
import frame_transformations as ft
import constants as const
import solver as solver
from source import Source
import numpy as np
import quaternion
import unittest



param=[101.28, -16.7161, 379.21, -546.05, -1223.14, 0]
t_init = 0
t_end =  365*5
my_dt = 1/24 # [days]
gaia = Satellite(ti=t_init, tf=t_end, dt= my_dt)
sirio = Source("sirio", 101.28, -16.7161, 379.21, -546.05, -1223.14, 0 )
A=sirio.compute_u(gaia, t_init)
print("A=",A)

p,q,r = ft.compute_pqr(101.28, -16.7161)
t_l=0
q_l = solver.attitude_from_alpha_delta(sirio,gaia,t_l,0)
B=sirio.compute_du_ds(gaia,p,q,r,q_l,t_l)
print ("B=", B)


#sirio= Source7p("sirio", 101.28, -16.7161, 379.21, -546.05, -1223.14, 0, 0, 0)
#t_l=0
#q_l = attitude_from_alpha_delta(sirio,gaia,t_l,0)
#C=sirio.compute_du_ds(gaia,p,q,r,q_l,t_l)
#print ("C=",C)

p,z=solver.observed_field_angles(sirio, q_l,  gaia, t_l, False)
print("phi=",p)
print("zeta=",z)
#pc,zc=calculated_field_angles(sirio, q_l,  gaia, t_l, False)
#print ("phi_calc=",pc)
#print ("zeta_calc=", zc)

m,n,u= solver.compute_mnu (p,z)
print ("m=",m)
T=B.transpose()
C=m@T*helpers.sec(z)
print("T=",T)
print ("C=",C)

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
