"""
Test suite for the hits module.


Toby James 2018
"""
import unittest
import numpy as np
import pandas as pd
import warnings
from array import array
from numba import NumbaWarning
import math

# Functions to run tests on
# Equivalent to from . import * but more verbose
try:
    from hits.hitdetector import identify_through_magnitude,\
        plot_anomaly, identify_through_gradient, Abuelmaatti, point_density, \
        filter_through_response, anomaly_density, rms_diff, stdev_diff, rms, \
        stdev
    from hits.hitsimulator import hit_distribution, flux, p_distribution, \
        freq, generate_event, generate_data, masses, tp_distribution, \
        time_distribution, AOCSResponse
except(ImportError):
    from .hitdetector import identify_through_magnitude, plot_anomaly, \
        identify_through_gradient, Abuelmaatti, point_density, \
        filter_through_response, anomaly_density, rms_diff, stdev_diff, rms, \
        stdev
    from .hitsimulator import hit_distribution, flux, p_distribution, freq, \
        generate_event, generate_data, masses, tp_distribution, \
        time_distribution, AOCSResponse


# -----------hitdetector.py tests----------------------------------------------
class TestHitDetectorIdentifyFuncs(unittest.TestCase):

    def setUp(self):
        # Create dummy data with anomalies to test for hits.
        obmt = np.linspace(0, 10, 1000)
        rate = np.zeros(1000)
        # Generate a random number of hits between 4 and 25.
        self.hits = np.random.randint(4, 25)

        hit_loc = np.linspace(2, 900, self.hits)
        for i in hit_loc:
            rate[int(i)] = 4

        w1_rate = np.zeros(1000)  # Value here is okay to be 0.
        self.df = pd.DataFrame(data=dict(obmt=obmt,
                                         rate=rate,
                                         w1_rate=w1_rate))

    def test_identify_through_magnitude_correctly_identifies(self):
        # Should identify 3 anomalies in the generated data.
        warnings.simplefilter("ignore", NumbaWarning)
        self.assertTrue(len(identify_through_magnitude(self.df)[1]) ==
                        self.hits)

    def test_identify_through_magnitude_return_shape(self):
        # Tests the function returns the expected dataframe shape.
        warnings.simplefilter("ignore", NumbaWarning)

        self.assertTrue(['obmt', 'rate', 'w1_rate', 'anomaly'] in
                        identify_through_magnitude(self.df)[0].columns.values)

    def test_identify_through_gradient_correctly_identifies(self):
        self.assertEqual(len(identify_through_gradient(self.df)[1]), self.hits,
                         msg="Detected %r hits. Expected to detect %r." %
                         (len(identify_through_gradient(self.df)[1]),
                          self.hits))

    def test_filter_through_anomaly_removes_all_anomalies(self):
        self.assertTrue(all(not x for x in
                            filter_through_response(self.df)['rate']),
                        msg="Anomalous data not filtered by "
                            "filter_by_response.")

    def test_filter_through_anomaly_doesnt_change_data_above_threshold(self):
        self.assertFalse(all(not x for x in
                             filter_through_response(self.df,
                                                     threshold=5)['rate']),
                         msg="Acceptable data filtered by filter_by_response.")

    def test_rms_diff_correctly_identifies_diff_of_1(self):
        data = np.zeros(1000)
        data[::2] += 1
        df = pd.DataFrame(data=dict(rate=data,
                                    w1_rate=np.zeros(1000)))
        diff = rms_diff(df)
        self.assertEqual(diff, 1,
                         msg="RMS diff for given data calculated as %r. "
                             "Expected 1." % diff)

    def test_stdev_diff_correctly_identifies_stdev_of_0(self):
        data = np.zeros(1000)
        data[::2] += 1
        df = pd.DataFrame(data=dict(rate=data,
                                    w1_rate=np.zeros(1000)))
        stdev = stdev_diff(df)
        self.assertEqual(0, stdev,
                         msg="stdev_diff calculated diff of %r. Expected 0.")

    def test_rms_correctly_identifies_rms_of_1(self):
        value = np.random.uniform(1, 20)
        data = np.ones(100) * value
        df = pd.DataFrame(data=dict(rate=data,
                                    w1_rate=np.zeros(100)))
        self.assertAlmostEqual(rms(df), value, places=6,
                               msg="rms identified rms value of %r. "
                                   "Expected %r." % (rms(df), value))

    def test_stdev_correctly_identifies_stdev_of_0(self):
        value = np.random.uniform(1, 20)
        data = np.ones(100) * value
        df = pd.DataFrame(data=dict(rate=data,
                                    w1_rate=np.zeros(100)))

        self.assertAlmostEqual(stdev(df), 0, msg="stdev identified standard "
                                                 "deviation value of %r. "
                                                 "Expected 0." % stdev(df))


class TestHitDetectorAbuelmaattiFuncs(unittest.TestCase):
    """
    The tests implemented here are based on the examples given in
    Abuelma'atti's original paper [1]. Values from the paper are tested
    against.

    Known values for periodic functions' periodicity are also tested
    against.

    [1] Abuelma'atti MT. A Simple Algorithm for Computing the Fourier
        Spectrum of Experimentally Obtained Signals. Applied Mathematics
        and Computation. 1999;98;pp229-239.
    """
    @staticmethod
    def func(i, t):
        return array('d', (max(a, 0) for a in np.sin(i * np.pi * t)))

    def setUp(self):
        self.time_array = np.linspace(0, 1, 1000)

        self.samples = self.func(2, self.time_array)

        self.a = Abuelmaatti(self.time_array, self.samples)

    def test_abuelmaatti_gamma_function_returns_05(self):

        self.assertAlmostEqual(self.a.gamma(1), 0.5, places=3,
                               msg="Calculated value for gamma(1) is 0.5. "
                               "Value returned was %r." % self.a.gamma(1))

    def test_abuelmaatti_gamma_function_returns_0(self):
        for i in range(1, 11):
            self.assertAlmostEqual(self.a.gamma(2), 0, places=3,
                                   msg="Calculated value for gamma(%r) is 0. "
                                   "Value returned was %r."
                                   % (i, self.a.gamma(2)))

    def test_abuelmaatti_delta_function_returns_expected_values(self):
        expected_deltas = [0, -0.212, 0, -0.042, 0, -0.018,
                           0, -0.0098, 0, -0.0064]
        delta_0 = 0.318

        self.assertAlmostEqual(self.a.delta_0, delta_0, places=3,
                               msg="Calculated value for delta_0 is %r. Value"
                               " returned was %r." % (delta_0, self.a.delta_0))
        for i in range(1, 11):
            self.assertAlmostEqual(self.a.delta(i), expected_deltas[i-1],
                                   places=3,
                                   msg="Calculated value for delta(%r) is %r."
                                   " Value returned was %r."
                                   % (i, expected_deltas[i-1],
                                      self.a.delta(i)))

    def test_abuelmaatti_returns_equal_values_for_regions_of_equal_phase(self):
        samples = self.func(1, self.time_array)
        a = Abuelmaatti(self.time_array[:int(len(samples)/2)],
                        samples[:int(len(samples)/2)])
        b = Abuelmaatti(self.time_array[:int(len(samples)/2)],
                        samples[int(len(samples)/2):])

        for i in range(1, 11):
            self.assertAlmostEqual(a.delta(i), b.delta(i), places=3,
                                   msg="%r and %r do not match for the %rth "
                                   "harmonic." % (a.delta(i), b.delta(i), i))


class TestHitDetectorDensityFuncs(unittest.TestCase):
    def setUp(self):
        point_array = np.linspace(1, 100, 100)
        self.df = pd.DataFrame(data=dict(rate=point_array,
                                         w1_rate=0))
        expected_density = np.array([10*[x]
                                     for x in range(1,
                                                    101)]).reshape(1, 1000)[0]

        self.expected_density = np.insert(expected_density, 0, 0)

    def test_point_density_returns_expected_height_array(self):
        try:
            np.testing.assert_array_almost_equal(point_density(self.df)[0],
                                                 np.arange(0, 100.1, 0.1),
                                                 decimal=3)
        except(AssertionError):
            raise(AssertionError("%r and %r are not equal."
                                 % (point_density(self.df)[0],
                                    np.arange(0, 100.1, 0.1))))

    def tests_point_density_returns_expected_density_array(self):
        try:
            np.testing.assert_array_almost_equal(point_density(self.df)[1],
                                                 self.expected_density,
                                                 decimal=3)
        except(AssertionError):
            raise(AssertionError("%r and %r are not equal."
                                 % ((point_density(self.df)[1]),
                                    (self.expected_density))))

    def test_anomaly_density_returns_correct_density(self):
        data = [0, 0, 3] * 100
        df = pd.DataFrame(data=dict(obmt=np.linspace(0, 100, 300),
                                    rate=data,
                                    w1_rate=[0] * 300))
        density = anomaly_density(df, window_size=3)
        try:
            np.testing.assert_array_almost_equal(density['density'],
                                                 [0] * 2 + [0.3333333] * 298,
                                                 decimal=3)
        except(AssertionError):
            raise(AssertionError("Density incorrectly calculated as %r."
                                 % density['density']))


# -----------hitsimulator.py tests---------------------------------------------
class TestHitSimulatorNumericalFuncs(unittest.TestCase):
    def test_hit_distribution_returns_correct_values(self):
        self.assertTrue(hit_distribution(1)[0][0] <= 2*np.pi)
        self.assertTrue(hit_distribution(1)[0][1] <= 4.5)

    def test_flux_expected_values_mass(self):
        # Test the flux function returns expected values.
        self.assertAlmostEqual(flux(2.7e-11), 5.388e-6, places=4,
                               msg="Flux of particles of mass 2.7e-11 "
                               "%r. Expected %r." %
                               (flux(2.7e-11), 5.388e-6))

        self.assertAlmostEqual(flux(2.9e-11), 5.114e-6, places=4,
                               msg="Flux of particles of mass 2.9e-11 "
                               "returned %r. Expected %r." %
                               (flux(2.9e-11), 5.114e-6))

    def test_freq_independent_of_mass_array(self):
        # The total frequency of a range should be the same irrespective
        # of the array of masses passed.
        masses2 = np.linspace(1e-13, 1e-7, 100)
        masses1_freq = sum(freq(masses))
        masses2_freq = sum(freq(masses2))
        self.assertAlmostEqual(masses1_freq, masses2_freq, places=4,
                               msg="Total frequency of 100 mass array "
                               "calculated as %r. Total frequency of "
                               "10000 mass array calculated as %r." %
                               (masses2_freq, masses1_freq))

    def test_p_distribution_returns_expected_shape(self):
        dist = p_distribution(np.linspace(0, 1, 1000))
        self.assertLessEqual(len(dist[1]), len(dist[0]),
                             msg="The array of hit indices was not less than"
                                 " the array of measured points.")

    def test_tp_distribution_returns_realistic_magnitude(self):
        for i in np.arange(0, 10, 0.05):
            points = tp_distribution(i)
            self.assertLess(points, 2*(i + 1),  # Big number
                            msg="%r turning points predicted for an input of"
                                "%r." % (points, i))
            self.assertTrue(points > 0,
                            msg="%r turning points predicted for an input of"
                                "%r. This value should be greater than 0,"
                                % (points, i))

    def test_time_distribution_returns_realistic_magnitude(self):
        for i in np.arange(0, 10, 0.05):
            time = time_distribution(i)
            self.assertLess(time, 0.1,  # Big number
                            msg="Response time was predicted to be %r for an "
                                "input of %r." % (time, i))
            self.assertTrue(time > 0,
                            msg="Response time was predicted to be %r for an "
                                "input of %r. This value should be greater "
                                "than 0." % (time, i))


class TestHitSimulatorAOCSClass(unittest.TestCase):

    def test_decay_pattern_length_increases_with_amplitude(self):
        len0p1 = len(AOCSResponse._decay_pattern(0.1))
        len100 = len(AOCSResponse._decay_pattern(100))
        self.assertLess(len0p1, len100, msg="AOCSResponse._decay_pattern "
                                            "returned arrays of length %r and "
                                            "%r for inputs of 0.1 and 100 "
                                            "respectively. 100 should yield a "
                                            "longer output than 0.1."
                                            % (len0p1, len100))

    def test_getitem_pops_value(self):
        data = AOCSResponse()
        data(12)
        len_before = len(data._data)
        data[0]
        len_after = len(data._data)

        self.assertEqual(len_before-1, len_after, msg="Calling __getitem__ "
                                                      "on a %r length instance"
                                                      " left the instance with"
                                                      " length %r. The length "
                                                      "should have decreased "
                                                      "by 1." % (len_before,
                                                                 len_after))

    def test_init_creates_empty_data_array(self):
        data = AOCSResponse()
        self.assertEqual(len(data._data), 0, msg="AOCSResponse instance "
                                                 "created with _data variable "
                                                 "of length %r. Length should "
                                                 "be 0." % len(data._data))

    def test_call_populates_data_array(self):
        data = AOCSResponse()
        data(12)
        self.assertNotEqual(len(data._data), 0, msg="AOCSResponse instance "
                                                    "has _data variable of "
                                                    "length 0. It should be "
                                                    "non-zero.")

    def test_call_with_small_amplitude_doesnt_increase_array_len(self):
        data = AOCSResponse()
        data(1000)
        big_len = len(data._data)
        data(0.5)
        new_len = len(data._data)

        self.assertEqual(big_len, new_len, msg="Calling an instance with "
                                               "length %r with value 0.5 "
                                               "increases the length to %r. "
                                               "The length shouldn't increase."
                                               % (big_len, new_len))


class TestHitSimulatorGeneratorFuncs(unittest.TestCase):

    @staticmethod
    def poisson_100(x):
        # Poisson distribution with rate parameter 1 - not exact but
        # indicative.
        return (np.e**(-10))/math.factorial(x)

    def test_expected_rate_of_impacts(self):
        # Test that generate_event creates expected rate of impact.
        # Conservative estimate used, expected rate is around 1% so it
        # is run 100 times and tested that the total amount is no
        # greater than 8. (Probability(X > 8 = 1e-6.)

        # Since this function is probabalistic it is difficult to test
        # for certain. It is therefore at the user's discretion (and
        # statistical understanding) to decide at what point a failure
        # truly indicates failure.
        frequencies = freq(masses)
        count = 0
        for _ in range(1000):
            # int(bool( to turn any non zero number into 1.
            count += 1*int(bool(generate_event(masses, frequencies)[0]))
        try:
            self.assertGreater(80, count)
        except(AssertionError):
            # Probability of receiving a value of count or higher.
            probability = 1 - sum([self.poisson_100(x) for x in range(count)])

            raise RuntimeError("\n\n***\nUnexpectedly high hitrate produced "
                               "by generate_data. Consider re-running tests."
                               "\n1000 attempts yielded %r hits. The "
                               "probability of this occurring is less than "
                               "%.2e.\nIf this happens multiple times, use "
                               "your statistical discretion to consider this "
                               "a failure.\n***\n\n" % (count, probability))

    def test_generate_data_returns_expected_length(self):
        data = generate_data(1000)
        self.assertEqual(len(data), 1000, msg="DataFrame produced was of "
                                              "length %r. Expected 1000."
                                              % len(data))

    def test_generate_data_doesnt_return_empty_array(self):
        data = generate_data(1000)
        self.assertFalse(all(bool(x) is False for x in data['rate']),
                         msg="Function generate_data generated no hits.")
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
