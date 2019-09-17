#This is the test for the 7 parameters Source model

import sys
 #append to path the folder that contains the analytic scanner
sys.path.append('../ESA/GaiaLab/gaialab')

import frame_transformations as ft
from scanner import Scanner
from satellite import Satellite
from source7p import Source
from source7p import Calc_source
import solver7p as solver

import constants as const
import quaternion
import helpers as helpers
import numpy as np

import astropy.units as units
import matplotlib.pyplot as plt

sirio = Source("sirio", 101.28, -16.7161, 379.21, -546.05, -1223.14, 0, 0, 0)
my_observations = [0,365]
calc_s =  Calc_source(obs_times=my_observations, source=sirio)
gaia = Satellite(0, 365*5, 1/24)
dR_ds_AL, dR_ds_AC, R_AL, R_AC, FA = solver.compute_design_equation(sirio,calc_s,gaia,my_observations)

alpha0 = calc_s.source.get_parameters()[0]
delta0 = calc_s.source.get_parameters()[1]
p, q, r = ft.compute_pqr(alpha0, delta0)
n_obs = len(my_observations)
dR_ds_AL = np.zeros((n_obs, 7))
for j, t_l in enumerate(my_observations):
    q_l = solver.attitude_from_alpha_delta(sirio,gaia,t_l,0)
    phi_obs, zeta_obs = solver.observed_field_angles(sirio, q_l,  gaia, t_l, False)
    phi_calc, zeta_calc = solver.calculated_field_angles(calc_s, q_l, gaia, t_l, False)
m,n,u= solver.compute_mnu (phi_calc,zeta_calc)
du_ds = sirio.compute_du_ds(gaia,p,q,r,q_l,t_l)
dR_ds_AL[j, :] = m @ du_ds.transpose() * helpers.sec(zeta_calc)

print("a=",alpha0)
print("d=",delta0)


print("dR_ds_AL=",dR_ds_AL)
#plt.scatter(FA[:,0]*units.rad.to(units.mas),FA[:,1]*units.rad.to(units.mas),c=my_observations)
#plt.show()
