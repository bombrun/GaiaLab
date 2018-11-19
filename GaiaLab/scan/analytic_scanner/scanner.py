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


def field_angles_per_FoV(source, attitude, sat, t, FoV='centered'):
    """
    :param FoV: specify which field of view we want. 'centered' if only one
    """
    Gamma_c = const.Gamma_c  # angle between the two telescopes

    Cu = source.unit_topocentric_function(sat, t)  # u in CoMRS frame
    Su = ft.lmn_to_xyz(attitude, Cu)
    Su_x, Su_y, Su_z = Su[:]

    phi = np.arctan2(Su_y, Su_x)
    zeta = np.arctan2(Su_z, np.sqrt(Su_x**2+Su_y**2))

    if FoV == 'centered':
        eta = phi
    elif FoV == 'following':
        eta = phi - Gamma_c / 2
    elif FoV == 'preceding':
        eta = phi + Gamma_c / 2
    else:
        raise ValueError('Invalid FoV parameter')
    return eta, zeta


def eta_error(t, sat, source):
    Cu_unit = source.unit_topocentric_function(sat, t)
    vector_error_xyz = ft.lmn_to_xyz(sat.func_attitude(t), Cu_unit) - np.array([1, 0, 0])
    return vector_error_xyz[1]  # WARNING: approx accurate only for small eta


def vector_error(t, sat, source):
    Cu_unit = source.unit_topocentric_function(sat, t)
    vector_error_xyz = ft.lmn_to_xyz(sat.func_attitude(t), Cu_unit) - np.array([1, 0, 0])
    return vector_error_xyz


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
    def scan(self, sat, source, ti=0, tf=5*const.days_per_year):
        """
        Find the exact time in which the source is seen.
        :param sat: [Satellite object]
        :param source: [Source object]
        :action: Find the observation time of the sources
        ideas to optimize: create array with all errors and go pick them as needed
        """
        print('Starting scan with time from {} to {} days'.format(ti, tf))
        self.reset()

        revolutions_per_day = sat.wz/(2*np.pi)
        time_of_revolution = 1/revolutions_per_day  # time in [days]
        time_step = time_of_revolution/6

        def violated_contraints(t_a, t_b, FoV):
            # eta_a, zeta_a = observed_field_angles(source, sat.func_attitude(t_a), sat, t_a, double_telescope)
            # eta_b, zeta_b = observed_field_angles(source, sat.func_attitude(t_b), sat, t_b, double_telescope)
            eta_a, zeta_a = field_angles_per_FoV(source, sat.func_attitude(t_a), sat, t_a, FoV)
            eta_b, zeta_b = field_angles_per_FoV(source, sat.func_attitude(t_b), sat, t_b, FoV)

            if eta_a*eta_b >= 0:  # check if f changes sign in [a,b]
                return True
            if np.abs(zeta_a + zeta_b)/2 > np.radians(0.5):   # ~ |zeta|<= 0.5°
                return True
            if (np.abs(eta_a) > np.pi/2) & (np.abs(eta_b) > np.pi/2):
                return True
            return False

        measured_time = 0
        # Looping
        for t in np.arange(ti, tf-time_step, time_step):
            # Check constraints
            t0 = time.time()
            for FoV in self.FoVs:
                if violated_contraints(t, t+time_step, FoV):
                    time_elapsed = time.time()-t0
                    measured_time += time_elapsed
                    continue
                x0, r = optimize.brentq(f=eta_angle, a=t, b=t+time_step, args=(sat, source, FoV),
                                        xtol=2e-20, rtol=8.881784197001252e-16,
                                        maxiter=100, full_output=True, disp=True)
                self.obs_times.append(x0)
                if self.FoVs == 'preceding':
                    self.obs_times_PFoV.append(x0)
                if self.FoVs == 'following':
                    self.obs_times_FFoV
                    self.obs_times_FFoV.append(x0)
                time_elapsed = time.time()-t0
                measured_time += time_elapsed
                print('time for constraints t:', x0, 'is', time_elapsed)
        print('Total measured time:', measured_time)
        # End of function

    def compute_angles_eta_zeta(self, sat, source):
        for t in self.obs_times:
            eta, zeta = observed_field_angles(source, sat.func_attitude(t), sat, t, self.double_telescope)
            # Cu_unit = source.unit_topocentric_function(sat, t)
            # Su = ft.lmn_to_xyz(sat.func_attitude(t), Cu_unit)
            # eta, zeta = compute_field_angles(Su, self.double_telescope)
            self.eta_scanned.append(eta)
            self.zeta_scanned.append(zeta)

    def scanner_error(self):
        if not self.eta_scanned:
            raise ValueError('No scanned angles in the scanner! Please scan' +
                             ' and call compute_angles_eta_zeta(sat, source)')
        return np.mean(self.eta_scanned)
# End of file
