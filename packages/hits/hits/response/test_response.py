import unittest
import numpy as np
import math
import pandas as pd

try:
    from hits.response.anomaly import isolate_anomaly, spline_anomaly
    from hits.response.characteristics import get_turning_points, \
        filter_turning_points, count_turning_points, response_time
except(ImportError):
    from .anomaly import isolate_anomaly, spline_anomaly
    from .characteristics import get_turning_points, filter_turning_points, \
        count_turning_points, response_time


# -----------characteristics.py tests------------------------------------------
class TestResponseTurningPointFuncs(unittest.TestCase):

    def setUp(self):
        # Set up dummy polynomial data with known number of turning
        # points.
        obmt = np.linspace(0, 100, 1000)
        # Random number of turning points between 0 and 20.
        self.points = np.random.randint(1, 20)

        # Set up rate as a sin function over obmt with the expected
        # number of turning points.
        rate = np.sin(self.points*np.pi*obmt/100)
        w1_rate = np.zeros(1000)
        self.df = pd.DataFrame(data=dict(obmt=obmt,
                                         rate=rate,
                                         w1_rate=w1_rate))

        hit_data = 100 * [0] + 100 * [3] + 100 * [0]
        self.hit_df = pd.DataFrame(data=dict(rate=hit_data,
                                             obmt=np.linspace(1, 300, 300),
                                             w1_rate=np.zeros(300),
                                             hits=100 * [False] +
                                             100 * [True] + 100 *
                                             [False]))

    def test_get_turning_points(self):
        got_points = len(get_turning_points(self.df))
        self.assertEqual(got_points, self.points,
                         msg="get_turning_points identified %r turning points."
                             " %r points expected." % (got_points,
                                                       self.points))

    def test_filter_turning_points(self):
        self.assertLessEqual(len(filter_turning_points(self.df)),
                             len(get_turning_points(self.df)),
                             msg="filter_turning_points returns more turning "
                                 "points than get_turning_points.")

    def test_count_turning_points_returns_correct_amplitude(self):
        measured_amp = count_turning_points(self.df)[0]
        self.assertAlmostEqual(measured_amp, 1, places=3,
                               msg="count_turning_points incorrectly "
                                   "identifies the peak as %r, rather than "
                                   "1." % measured_amp)

    def test_count_turning_points_counts_correct_number_of_points(self):
        counted_points = count_turning_points(self.df, threshold=0.4)[1]
        self.assertEqual(counted_points, self.points - 1,
                         msg="count_turning_points returned %r points. "
                             "Expected to return %r." % (counted_points,
                                                         self.points - 1))

    def test_response_time_returns_correct_response_time(self):
        time = response_time(self.hit_df)[1]
        self.assertEqual(time, 101,
                         msg="response_time returned %r. Expected 101." % time)

    def test_response_time_returns_correct_peak(self):
        peak = response_time(self.hit_df)[0]
        self.assertEqual(peak, 3,
                         msg="response_time identified a peak of %r. Expected"
                             " 3." % peak)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
