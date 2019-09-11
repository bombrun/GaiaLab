#This is the test for the 7 parameters Source model

import sys
 #append to path the folder that contains the analytic scanner
sys.path.append('../ESA/GaiaLab/gaialab')

import frame_transformations as ft
from scanner import Scanner
from satellite import Satellite
from source7p import Source
from source7p import Calc_source
from solver7p import compute_design_equation
import solver7p

import constants as const
import quaternion
import agis_functions as af
import helpers as helpers
import numpy as np

import astropy.units as units
import matplotlib.pyplot as plt

sirio = Source("sirio", 101.28, -16.7161, 379.21, -546.05, -1223.14, 0, 0, 0)
my_observations = np.arange(0,365*3,15)
calc_s =  Calc_source(obs_times=my_observations, source=sirio)
gaia = Satellite(0, 365*5, 1/24)
dR_ds_AL, dR_ds_AC, R_AL, R_AC, FA = compute_design_equation(sirio,calc_s,gaia,my_observations)

plt.scatter(FA[:,0]*units.rad.to(units.mas),FA[:,1]*units.rad.to(units.mas),c=my_observations)
plt.show()
print("ok")
