"""
Scanner class implementation in Python


@author: mdelvallevaro
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

    def reset_memory(self):
        """
        :action: empty all attribute lists from scanner before beginning new scanning period.
        """
        self.obs_times.clear()
        self.root_messages.clear()

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
        self.reset_memory()
        time_step = 1/24

        # fonctions used in the loop
        def eta_got(t):
            Cu_unit = source.unit_topocentric_function(sat, t)
            vector, angle = helpers.get_rotation_vector_and_angle(v1=Cu_unit, v2=sat.func_x_axis_lmn(t))
            angle = angle % (2*np.pi)
            if angle > np.pi:
                angle = - np.pi + (angle - np.pi)  # set [0, 2pi] to [-pi, pi]
            return angle

        def eta_error(t):
            Cu_unit = source.unit_topocentric_function(sat, t)
            vector_error_xyz = ft.lmn_to_xyz(sat.func_attitude(t), Cu_unit) - np.array([1, 0, 0])
            return vector_error_xyz[1]  # WARNING: approx accurate only for small eta

        def vector_error(t):
            Cu_unit = source.unit_topocentric_function(sat, t)
            vector_error_xyz = ft.lmn_to_xyz(sat.func_attitude(t), Cu_unit) - np.array([1, 0, 0])
            return vector_error_xyz

        def violated_contraints(t_a, t_b):
            bool_violated_constraints = False
            if np.abs(vector_error(t_a)[2] + vector_error(t_b)[2]) > np.radians(0.5):
                bool_violated_constraints = True
            if (np.abs(eta_got(t_a)) > np.pi/2) & (np.abs(eta_got(t_b)) > np.pi/2):
                bool_violated_constraints = True
            if eta_error(t_a)*eta_error(t_b) >= 0:  # check if f changes sign in [a,b]
                bool_violated_constraints = True
            return bool_violated_constraints
        # Looping
        measured_time = 0
        for t in np.arange(ti, tf-time_step, time_step):
            # Check constraints
            t0 = time.time()
            if violated_contraints(t, t+time_step):
                time_elapsed = time.time()-t0
                measured_time += time_elapsed
                # print('time for constraints t:', t, 'is', time_elapsed)
                continue
            time_elapsed = time.time()-t0
            measured_time += time_elapsed
            print('time for constraints t:', t, 'is', time_elapsed)
            t0 = time.time()
            x0, r = optimize.brentq(f=eta_error, a=t, b=t+time_step, args=(),
                                    xtol=2e-20, rtol=8.881784197001252e-16,
                                    maxiter=100, full_output=True, disp=True)
            time_elapsed = time.time()-t0
            measured_time += time_elapsed
            print('time for t:', t, '  :', time_elapsed)
            self.obs_times.append(x0)
        print('Total measured time:', measured_time)
        # End of function







# End of file
