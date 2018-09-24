# # Constant storage file
#
# File containing the constants that will be used in the other files. This will
# allow to avoid "magic numbers" in the code and also to easily change these
# constants if we later need them more or less precises
#
# Luca Zampieri 2018

import numpy as np

# General
days_per_year = 365  # [days]
rad_per_mas = 2*np.pi/(1000*360*3600)  # [radiants] radiants per milli-arcsec
km_per_pc = 3.24078e-14  # [km ]kilometers per parsec
sec_per_day = 3600*24  # [sec] seconds per day
AU_per_pc = 4.8481705933824e-6  # [au] austronomical unit per parsec

# Proper to Gaia
Gamma_c = 106.5  # [deg] basic angle = arccos(f_p' f_F)
