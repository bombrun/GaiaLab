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
import solver7p as solver
from source7p import Source
import numpy as np
import quaternion
import unittest

t_init = 0
t_end =  365*5
my_dt = 1/24 # [days]
gaia = Satellite(ti=t_init, tf=t_end, dt= my_dt)

sirio = Source("sirio", 101.28, -16.7161, 379.21, -546.05, -1223.14, 0, 0, 0 )
t_l=0
q_l = solver7p.attitude_from_alpha_delta(sirio,gaia,t_l,0)
p,z=solver7p.observed_field_angles(sirio, q_l,  gaia, t_l, False)
#print("phi=",p)
#print("zeta=",z)
#m,n,u= compute_mnu (p,z)
#print ("m=",m)
p,q,r = ft.compute_pqr(101.28, -16.7161)
t_l=0
q_l = solver.attitude_from_alpha_delta(sirio,gaia,t_l,0)
B=sirio.compute_du_ds(gaia,p,q,r,q_l,t_l)
print ("B=", B)

p,z=solver.observed_field_angles(sirio, q_l,  gaia, t_l, False)
print("phi=",p)
print("zeta=",z)

m,n,u= solver.compute_mnu (p,z)
print ("m=",m)
T=B.transpose()
C=m@T*helpers.sec(z)
print("T=",T)
print ("C=",C)

my_observations=[0,365]
calc_s =  Calc_source(obs_times=my_observations, source=sirio)
dR_ds_AL, dR_ds_AC, R_AL, R_AC, FA = solver7p.compute_design_equation(sirio,calc_s,gaia,my_observations)
print("dR_ds_AL",dR_ds_AL)
