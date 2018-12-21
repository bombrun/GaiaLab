# -*- coding: utf-8 -*-
"""
Scanner class implementation in Python

:Authors: LucaZampieri (2018)
         mdelvallevaro (2018)
"""

# # Imports
# Global imports
import numpy as np
import time
from scipy import optimize
import quaternion
# Local imports
import constants as const
import frame_transformations as ft
from satellite import Satellite
from source import Source
import helpers
from agis_functions import *


# fonctions used in the loop
def eta_angle(t, sat, source, FoV='centered'):
    """
    | Function to minimize in the scanner.
    | See :meth:`agis_functions.observed_field_angles`

    :param FoV: [string] specify which Field of View to use
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
    """
    Tranform phis into etas using the field of view parameter

    :param phi_a: phi at the beginning of the interval
    :param phi_b: phi at the end of the interval
    :param FoV: [string] specify which Field of View we are using
    :returns: eta at the beginning and end of the inteval
    """

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
    """
    :returns: True if the contraints are violated and False otherwise
    """
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
        :param zeta_limit: [rads] limitation of the Field of View in the across scan direction
        :param double_telescope: [bool] if true implements the scanner version with two
         telescopes (gaia-like)
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
        :action: empty all attribute lists from scanner before beginning new
         scanning period.
        :param verbose: [bool] If true will print messages
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

        :action: Find the observation time of the sources
        :param sat: [Satellite object]
        :param source: [Source object]
        :param ti & tf: [days] initial and end dates
        :returns: [float] time it took for the scan
        """
        # print('Starting scan with time from {} to {} days'.format(ti, tf))
        self.reset()
        t0 = time.time()  # for timer

        time_step = sat.time_of_revolution/6  # need <= 6th of revolution time

        # Get list on which to loop
        if (tf - ti) > 10:
            day_list = get_interesting_days(ti, tf, sat, source, self.zeta_limit)
            t_list = generate_scanned_times_intervals(day_list, time_step)
        else:
            t_list = np.arange(ti, tf, time_step)
        t_old = 0
        # Looping
        for t in t_list:
            # Check constraints
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
                                        xtol=2e-20, rtol=8.881784197001252e-16,
                                        maxiter=100, full_output=True, disp=False)
                self.obs_times.append(x0)
                """if FoV == 'preceding':
                    self.obs_times_PFoV.append(x0)
                elif FoV == 'following':
                    self.obs_times_FFoV.append(x0)
                else:
                    pass"""
            t_old = t+time_step
        # self.obs_times = list(np.sort(self.obs_times))
        return time.time()-t0  # Total measured time

    def compute_angles_eta_zeta(self, sat, source):
        """
        Compute angles and remove 'illegal' observations (:math:`|zeta| > zeta_lim`)
        """
        for t in self.obs_times:
            eta, zeta = observed_field_angles(source, sat.func_attitude(t), sat, t, self.double_telescope)
            if np.abs(zeta) >= self.zeta_limit:
                continue
            self.eta_scanned.append(eta)
            self.zeta_scanned.append(zeta)

    def scanner_error(self):
        """
        :return: mean error in the Along-scan direction
        """
        if not self.eta_scanned:
            raise ValueError('No scanned angles in the scanner! Please scan' +
                             ' and call compute_angles_eta_zeta(sat, source)')
        return np.mean(self.eta_scanned)
# End of file
