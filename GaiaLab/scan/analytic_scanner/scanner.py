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


class Scanner:

    def __init__(self, wide_angle=np.radians(20),  scan_line_height=np.radians(5)):
        """
        :param wide_angle: angle for first dot-product-rough scan of all the sky.
        :param scan_line_height: condition for the line_height of the scanner (z-axis height in lmn)

        Attributes
        -----------
        :coarse_angle: uses scan_line_height to get measurements after wide_angle dot product. [rad]
        :times_wide_scan: times where star in wide_angle field of view. [days]
        :times_coarse_scan: times where star in wide_angle and coarse angle field of view [days]
        :optimize_roots: [func] minimized solutions
        :roots: [func] found roots
        :obs_times: the times of the roots (i.e. self.roots.x) where the star precisely crosses the line of view [days]
        """
        self.wide_angle = wide_angle  # Threshold for the x-axis
        self.scan_line_height = scan_line_height/2.  # Threshold over z axis
        self.coarse_angle = self.scan_line_height  # angle of view of the coarse scan

        self.z_threshold = np.sin(self.scan_line_height)

        self.times_wide_scan = []
        self.times_coarse_scan = []

        self.times_optimize = []
        self.optimize_roots = []
        self.roots = []
        self.obs_times = []

    def reset_memory(self):
        """
        :return: empty all attribute lists from scanner before beginning new scanning period.
        """
        self.times_wide_scan.clear()
        self.times_coarse_scan.clear()

        self.times_optimize.clear()
        self.optimize_roots.clear()
        self.roots.clear()
        self.obs_times.clear()

    def start(self, sat, source, ti=0, tf=5*const.days_per_year):
        print('Starting wide_scan with time from {} to {} days'.format(ti, tf))
        self.wide_scan(sat, source, ti, tf)
        print('Finished wide_scan!')

        self.coarse_scan(sat, source, ti, tf)
        print('Finished coarse_scan!')

        self.fine_scan(sat, source)
        print('Finished fine_scan!')

    def wide_scan(self, sat, source, ti=0, tf=5*const.days_per_year):
        """
        Scans sky with a dot product technique to get rough times of observation.
        :action: self.times_wide_scan list filled with observation time windows.
        """
        # if not isinstance(sat, satellite):
        #    raise TypeError('Expected Satellite, but got {} instead'.format(type(sat)))
        # if not isinstance(source, Source):
        #    raise TypeError('Expected Source, but got {} instead'.format(type(source)))

        # Reset the memory of the previous scans
        self.reset_memory()

        t_0 = time.time()  # t0 of the timer

        self.step_wide = self.wide_angle / (2 * np.pi * 4)
        for t in np.arange(ti, tf, self.step_wide):
            to_star_unit = source.unit_topocentric_function(sat, t)
            angle_source_xaxis = np.arccos(np.dot(to_star_unit, sat.func_x_axis_lmn(t)))
            if angle_source_xaxis < self.wide_angle:
                self.times_wide_scan.append(t)
        time_wide = time.time()  # time after wide scan
        print('wide scan lasted {} seconds, found {} times with wide scan'
              .format(time_wide - t_0, len(self.times_wide_scan)))

        # # Alternative way to do it:
        # my_ts = np.arange(ti, tf, step_wide)
        # def f(x):
        #     to_star_unit = source.unit_topocentric_function(sat, x)
        #     return np.arccos(np.dot(to_star_unit, sat.func_x_axis_lmn(x))) < self.wide_angle
        # array_map = np.array(list(map(f, my_ts)))
        # self.times_wide_scan = list(my_ts[np.nonzero(array_map)])
        # #

    def coarse_scan(self, sat, source, ti=0, tf=5*const.days_per_year):
        t_0 = time.time()  # reset the  t_0 of the time
        # Make the coarse angle scan
        step_coarse = self.coarse_angle / (2 * np.pi * 4)
        for t_wide in self.times_wide_scan:
            for t in np.arange(t_wide - self.step_wide / 2, t_wide + self.step_wide / 2, step_coarse):
                to_star_unit = source.unit_topocentric_function(sat, t)
                if np.arccos(np.dot(to_star_unit, sat.func_x_axis_lmn(t))) < self.coarse_angle:
                    self.times_coarse_scan.append(t)
        time_coarse = time.time()  # time after coarse scan
        print('Coarse scan lasted {} seconds, found {} times with coarse scan'
              .format(time_coarse - t_0, len(self.times_coarse_scan)))

    # fine_scan function
    def fine_scan(self, sat, source, tolerance=1e-3):
        """
        Find the exact time in which the source is seen. Only the times when the
        source is in the field of view are scanned, i.e. self.times_coarse_scan.
        :param sat: [Satellite object]
        :param source: [Source object]
        :param tolerance: [int,float, optional] [days] tolerance up to which we distinguish
            two observations
        :action: Find the observation time of the sources
        """

        def phi_objective(t):
            t = float(t)
            u_lmn_unit = source.unit_topocentric_function(sat, t)
            phi_vector_lmn = u_lmn_unit - sat.func_x_axis_lmn(t)
            phi_vector_xyz = ft.lmn_to_xyz(sat.func_attitude(t), phi_vector_lmn)
            return np.abs(phi_vector_xyz[1])

        def z_condition(t):
            t = float(t)
            u_lmn_unit = source.unit_topocentric_function(sat, t)
            phi_vector_lmn = u_lmn_unit - sat.func_x_axis_lmn(t)
            phi_vector_xyz = ft.lmn_to_xyz(sat.func_attitude(t), phi_vector_lmn)
            z_threshold = np.sin(self.scan_line_height)
            return z_threshold - np.abs(phi_vector_xyz[2])  # >= 0 for scipy.optimize.minimize

        con1 = {'type': 'ineq', 'fun': z_condition}  # inequality constraint: z_condition >= 0

        time_step = self.coarse_angle / (2 * np.pi * 4)
        print('time_step: {}'.format(time_step))

        t_0 = time.time()  # set t_0 of the timer
        # find times where possible solutions are
        for i in self.times_coarse_scan:
            def t_condition(t):
                if i - time_step < t < i + time_step:
                    return 1.0
                else:
                    return -1.0

            con2 = {'type': 'ineq', 'fun': t_condition}

            optimize_root = optimize.minimize(phi_objective, i, method='COBYLA', constraints=[con1, con2])
            if optimize_root.success:
                self.times_optimize.append(float(optimize_root.x))
                self.optimize_roots.append(optimize_root)
                # TODO: optimize the z-coordinate
                # optimize_z = optimize.minimize(z, i, method='COBYLA', constraints=[con1, con2])
        time_optimize_root = time.time()  # time after wide scan
        print('phi_minimization lasted {} seconds'.format(time_optimize_root - t_0))

        t_0 = time.time()  # reset t_0 of the timer
        # find roots for phi
        for obj in self.optimize_roots:
            root = optimize.root(phi_objective, [obj.x])
            self.roots.append(root)
            self.obs_times.append(float(root.x))
        time_phi_root = time.time()  # time after wide scan
        print('wide scan lasted {} seconds'.format(time_phi_root - t_0))

        # remove identical duplicates
        print('original obs_times: {}'.format(len(self.obs_times)))
        self.obs_times = list(set(self.obs_times))
        self.obs_times.sort()  # to leave them in increasing order
        print('identical duplicates removal obs_time: {}'.format(len(self.obs_times)))
