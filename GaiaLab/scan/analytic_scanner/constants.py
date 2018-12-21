# -*- coding: utf-8 -*-
"""
:File: constants.py

:Purpose: File containing the constants that will be used in the other files. This will
 allow to avoid "magic numbers" in the code and also to easily change these
 constants if we later need them more or less precises

:Author: Luca Zampieri 2018

"""

# # Imports
import numpy as np

#: General
#: How many days per year
days_per_year = 365  # [days/years]
#: How many radians per micro arcsec
rad_per_mas = 2*np.pi/(1000*360*3600)  # [radiants/mas] radiants per milli-arcsec
rad_per_arcsec = 2*np.pi/(360*3600)  # [radiants/arcsec] radiants per arcsec
rad_per_deg = (2*np.pi)/360  # [radiants/degrees]
pc_per_km = 3.24078e-14  # [km/pc] kilometers per parsec
sec_per_day = 3600*24  # [sec/day] seconds per day
km_per_Au = 149598000  # number of kilometers in one Au
AU_per_pc = 4.8481705933824e-6  # [au/pc] austronomical unit per parsec
Au_per_km = 1/km_per_Au
c = 299.792458e6  # [m/s]
Au_per_Au = 1  # useless, just to make computations explicit?


# # Proper to Gaia
# constant specific to gaia that have been chosen. (see e.g. )
epsilon = 23 + 26/60 + 21.448/3600  # [deg] obiquity of equator chosen to be 23º 26' 21.448''
Gamma_c = np.radians(106.5)  # [rad] basic angle, Gamma_c = arccos(f_p' f_F)
xi = 55  # [deg] angle between the z-axis and s (s being the nominal sun direction)
S = 4.035  # [deg/day] for a xi of 55°. S=|dz/dlambda|
w_z = 60  # [arcsec/s] z component of the inertial spin vector w (small omega)
#: Epoch time
#: The reference epoch is J2000 but it is taken into account in how we count time thus t_ep is 0
t_ep = 0

# temporary
sat_angle = np.radians(45)  # when simulating the attitude, the rotated angle


def useless_function():
    """
    This function does nothing. Here only for testing purpose.
    """
    pass
