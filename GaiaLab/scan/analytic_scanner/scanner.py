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
def phi_angle(sat, source, t):
    Cu_unit = source.unit_topocentric_function(sat, t)
    Su = ft.lmn_to_xyz(sat.func_attitude(t), Cu_unit)
    """
    Su_xy = [Su[0], Su[1], 0]  # projection on the xy-plane
    vector, angle = helpers.get_rotation_vector_and_angle(v1=Su_xy, v2=[1, 0, 0])
    angle = angle % (2*np.pi)
    if angle > np.pi:
        angle = - np.pi + (angle - np.pi)  # set [0, 2pi] to [-pi, pi]
    return angle"""
    Su_x, Su_y, Su_z = Su[:]
    phi = np.arctan2(Su_y, Su_x)
    return phi


def eta_error(t, sat, source):
    Cu_unit = source.unit_topocentric_function(sat, t)
    vector_error_xyz = ft.lmn_to_xyz(sat.func_attitude(t), Cu_unit) - np.array([1, 0, 0])
    return vector_error_xyz[1]  # WARNING: approx accurate only for small eta


def vector_error(sat, source, t):
    Cu_unit = source.unit_topocentric_function(sat, t)
    vector_error_xyz = ft.lmn_to_xyz(sat.func_attitude(t), Cu_unit) - np.array([1, 0, 0])
    return vector_error_xyz


class Scanner:

    def __init__(self,  zeta_limit=np.radians(0.5)):
        """
        :param wide_angle: angle for first dot-product-rough scan of all the sky.
        :param zeta_limit: condition for the line_height of the scanner (z-axis height in lmn)

        Attributes
        -----------
        :obs_times: [days] the times where the star precisely crosses the line
                    of view (time for which the error in eta is minimal)
        :root_messages:
        """
        self.zeta_limit = zeta_limit

        # Products
        self.obs_times = []
        self.root_messages = []
        self.eta_scanned = []
        self.zeta_scanned = []

    def reset(self, verbose=False):
        """
        :action: empty all attribute lists from scanner before beginning new scanning period.
        """
        self.obs_times.clear()
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
        time_of_revolution = 1/revolutions_per_day
        time_step = time_of_revolution/3

        def violated_contraints(t_a, t_b):
            bool_violated_constraints = False
            # checks zeta constraint
            if np.abs(vector_error(sat, source, t_a)[2] + vector_error(sat, source, t_b)[2]) > np.radians(0.5):
                bool_violated_constraints = True
            # check about angles
            if (np.abs(phi_angle(sat, source, t_a)) > np.pi/2) & (np.abs(phi_angle(sat, source, t_b)) > np.pi/2):
                bool_violated_constraints = True
            # check if f changes sign in [a,b]
            if eta_error(t_a, sat, source)*eta_error(t_b, sat, source) >= 0:
                bool_violated_constraints = True
            return bool_violated_constraints
        # Looping
        measured_time = 0
        for t in np.arange(ti, tf-time_step, time_step):
            # Check constraints
            t0 = time.time()
            if violated_contraints(t, t+time_step):
                measured_time += time.time()-t0
                continue
            time_elapsed = time.time()-t0
            measured_time += time_elapsed
            print('time for constraints t:', t, 'is', time_elapsed)
            t0 = time.time()
            x0, r = optimize.brentq(f=eta_error, a=t, b=t+time_step, args=(sat, source),
                                    xtol=2e-20, rtol=8.881784197001252e-16,
                                    maxiter=100, full_output=True, disp=True)
            time_elapsed = time.time()-t0
            measured_time += time_elapsed
            print('time for t:', t, '  :', time_elapsed)
            self.obs_times.append(x0)
        print('Total measured time:', measured_time)
        # End of function

    def compute_angles_eta_zeta(self, sat, source):
        for t in self.obs_times:
            Cu_unit = source.unit_topocentric_function(sat, t)
            Su = ft.lmn_to_xyz(sat.func_attitude(t), Cu_unit)
            eta, zeta = compute_field_angles(Su)
            self.eta_scanned.append(eta)
            self.zeta_scanned.append(zeta)

    def scanner_error(self):
        if not self.eta_scanned:
            raise ValueError('No scanned angles in the scanner! Please scan' +
                             ' and compute the angles')
        return np.mean(self.eta_scanned)


# End of file
