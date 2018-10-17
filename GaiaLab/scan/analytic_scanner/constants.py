"""
This file stores the constants used, in order to avoid magic numbers in the code

File containing the constants that will be used in the other files. This will
allow to avoid "magic numbers" in the code and also to easily change these
constants if we later need them more or less precises

author:: Luca Zampieri 2018
"""

# # Imports
import numpy as np

# General
days_per_year = 365  # [days/years]
rad_per_mas = 2*np.pi/(1000*360*3600)  # [radiants/mas] radiants per milli-arcsec
km_per_pc = 3.24078e-14  # [km/pc] kilometers per parsec
sec_per_day = 3600*24  # [sec/day] seconds per day
AU_per_pc = 4.8481705933824e-6  # [au/pc] austronomical unit per parsec
c = 299.792458e6  # [m/s]
# km_per_au = 149597870.700  # [km/au]
# pc_per_au = 4.8481e-6

# # Proper to Gaia
# constant specific to gaia that have been chosen. (see e.g. )
epsilon = 23 + 26/60 + 21.448/3600  # [deg] obiquity of equator chosen to be 23º 26' 21.448''
Gamma_c = 106.5  # [deg] basic angle, Gamma_c = arccos(f_p' f_F)
xi = 55  # [deg] angle between the z-axis and s (s being the nominal sun direction)
S = 4.035  # [deg/day] for a xi of 55°. S=|dz/dlambda|
w_z = 120  # [arcsec/s] z component of the inertial spin vector w (small omega)
# The reference epoch is J2000 but it is taken into account in how we count time thus t_ep is 0
t_ep = 0  # epoch time


def test__():
    """
    This is a test
    """
    pass
