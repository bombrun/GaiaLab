"""
Scanner class implementation in Python

@author: LucaZampieri
"""

# # Imports
# Global imports
import numpy as np
import time
from scipy import optimize
# Local imports
import constants as const
import frame_transformations as ft
from quaternion import Quaternion
from satellite import Satellite
from source import Source
import helpers
from agis_functions import *


# fonctions used in the loop
def eta_angle(t, sat, source, FoV='centered'):
    """
    Function to minimize in the scanner.
    """
    Gamma_c = const.Gamma_c

    Cu_unit = source.unit_topocentric_function(sat, t)
    Su = ft.lmn_to_xyz(sat.func_attitude(t), Cu_unit)

    Su_x, Su_y, Su_z = Su[:]
    phi = np.arctan2(Su_y, Su_x)
    field_index = np.sign(phi)
    if FoV == 'centered':
        eta = phi
    elif FoV == 'preceding':
        eta = phi + Gamma_c / 2
    elif FoV == 'following':
        eta = phi - Gamma_c / 2
    else:
        raise ValueError('incorrect FoV argument.')
    return eta


def get_etas_from_phis(phi_a, phi_b, FoV):
    Gamma_c = const.Gamma_c
    if FoV == 'following':
        eta_a, eta_b = (phi_a - Gamma_c / 2, phi_b - Gamma_c / 2)
    elif FoV == 'preceding':
        eta_a, eta_b = (phi_a + Gamma_c / 2, phi_b + Gamma_c / 2)
    elif FoV == 'centered':
        eta_a, eta_b = (phi_a, phi_b)
    else:
        raise ValueError('Invalid FoV parameter')
    return eta_a, eta_b


def violated_contraints(eta_a, zeta_a, eta_b, zeta_b, zeta_limit):
    if eta_a*eta_b >= 0:  # check if f changes sign in [a,b]
        return True
    if np.abs(zeta_a + zeta_b)/2 > zeta_limit:   # ~ |zeta|<= 0.5°
        return True
    if (np.abs(eta_a) > np.pi/2) & (np.abs(eta_b) > np.pi/2):
        return True
    return False


# Scanner class
class Scanner:

    def __init__(self,  zeta_limit=np.radians(0.5), double_telescope=True):
        """
        :param zeta_limit: condition for the line_height of the scanner (z-axis height in lmn)
        Attributes
        -----------
        :obs_times: [days] the times where the star precisely crosses the line
                    of view (time for which the error in eta is minimal)
        :root_messages:
        """
        # Parameters
        self.zeta_limit = zeta_limit
        self.double_telescope = double_telescope  # bool that indicates if there is the second telescope
        self.FoVs = ['preceding', 'following'] if double_telescope else ['centered']  # fields of view of scanner

        # Products
        self.obs_times = []
        self.obs_times_FFoV = []
        self.obs_times_PFoV = []

        self.root_messages = []
        self.eta_scanned = []
        self.zeta_scanned = []

    def reset(self, verbose=False):
        """
        :action: empty all attribute lists from scanner before beginning new scanning period.
        """
        self.obs_times.clear()
        self.obs_times_FFoV.clear()
        self.obs_times_PFoV.clear()
        self.root_messages.clear()
        self.eta_scanned.clear()
        self.zeta_scanned.clear()

        if verbose:
            print('Cleared variables!')

    # Scan function
    def scan(self, sat, source, ti, tf):
        """
        Find the exact time in which the source is seen.
        :param sat: [Satellite object]
        :param source: [Source object]
        :param ti & tf: [days] initial and end dates
        :action: Find the observation time of the sources
        ideas to optimize: create array with all errors and go pick them as needed
        """
        print('Starting scan with time from {} to {} days'.format(ti, tf))
        self.reset()
        t0 = time.time()  # for timer

        time_step = sat.time_of_revolution/12  # need <= 6th of revolution time

        # Get list on which to loop
        day_list = get_interesting_days(ti, tf, sat, source)
        t_list = generate_scanned_times_intervals(day_list, time_step)

        t_old = 0
        # Looping
        for t in np.arange(ti, tf-time_step, time_step):
        # for t in t_list:
            # Check constraints
            # print(t)
            if (t == t_old) & (t_old > 0):
                # print(t)
                phi_a, zeta_a = (phi_b, zeta_b)
            else:
                phi_a, zeta_a = observed_field_angles(source, sat.func_attitude(t), sat, t, double_telescope=False)
            phi_b, zeta_b = observed_field_angles(source, sat.func_attitude(t+time_step), sat, t+time_step,
                                                  double_telescope=False)

            for FoV in self.FoVs:
                eta_a, eta_b = get_etas_from_phis(phi_a, phi_b, FoV)
                if violated_contraints(eta_a, zeta_a, eta_b, zeta_b, self.zeta_limit):
                    continue
                x0, r = optimize.brentq(f=eta_angle, a=t, b=t+time_step, args=(sat, source, FoV),
                                        xtol=2e-16, rtol=8.881784197001252e-16,
                                        maxiter=100, full_output=True, disp=False)
                self.obs_times.append(x0)
            t_old = t+time_step

        print('Total measured time:', time.time()-t0)
        # End of function

    def compute_angles_eta_zeta(self, sat, source):
        """ Compute angles and remove 'illegal' observations (|zeta| > zeta_lim)"""
        for t in self.obs_times:
            eta, zeta = observed_field_angles(source, sat.func_attitude(t), sat, t, self.double_telescope)
            if np.abs(zeta) >= self.zeta_limit:
                continue
            self.eta_scanned.append(eta)
            self.zeta_scanned.append(zeta)

    def scanner_error(self):
        if not self.eta_scanned:
            raise ValueError('No scanned angles in the scanner! Please scan' +
                             ' and call compute_angles_eta_zeta(sat, source)')
        return np.mean(self.eta_scanned)
# End of file
