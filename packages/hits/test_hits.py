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

#functions to run tests on
#equivalent to from . import * but more verbose
try:
    from hits.hitdetector import identify_through_magnitude,\
                                 plot_anomaly, identify_through_gradient, \
                                 Abuelmaatti, point_density
    from hits.hitsimulator import hit_distribution, flux, p_distribution, \
                                  freq, generate_event, generate_data, masses
    from hits.response.anomaly import isolate_anomaly, spline_anomaly
    from hits.response.characteristics import get_turning_points, \
                                              filter_turning_points
except(ImportError):
    from .hitdetector import identify_through_magnitude, plot_anomaly, \
                             identify_through_gradient, Abuelmaatti, \
                             point_density
    from .hitsimulator import hit_distribution, flux, p_distribution, freq, \
                              generate_event, generate_data, masses
    from .response.anomaly import isolate_anomaly, spline_anomaly
    from .response.characteristics import get_turning_points, \
                                          filter_turning_points


#------------hitdetector.py tests----------------------------------------------
class TestHitDetectorIdentifyFuncs(unittest.TestCase):

    def setUp(self):
        # Create dummy data with anomalies to test for hits.
        obmt = np.linspace(0,10,1000)
        rate = np.zeros(1000)
        # Generate a random number of hits between 4 and 25.
        self.hits = np.random.randint(4,25)
        
        hit_loc = np.linspace(2,900, self.hits)
        for i in hit_loc:
            rate[int(i)] = 4

        w1_rate = np.zeros(1000) # Value here is okay to be 0.
        
        self.df = pd.DataFrame(data=dict(obmt = obmt,
                                         rate = rate,
                                         w1_rate = w1_rate))
        sin_data = 3*np.sin(2*np.pi * obmt)
        self.sin_df = pd.DataFrame(data=dict(obmt = obmt,
                                             rate = sin_data,
                                             w1_rate = w1_rate))
    def test_identify_through_magnitude_correctly_identifies(self):
        # Should identify 3 anomalies in the generated data.
        warnings.simplefilter("ignore", NumbaWarning)
        self.assertTrue(len(identify_through_magnitude(self.df)[1]) == self.hits)

    def test_identify_through_magnitude_return_shape(self):
        # Tests the function returns the expected dataframe shape.
        warnings.simplefilter("ignore", NumbaWarning)
        
        self.assertTrue(['obmt', 'rate', 'w1_rate', 'anomaly'] in \
        identify_through_magnitude(self.df)[0].columns.values)

    def test_identify_through_gradient_correctly_identifies(self):
        
        self.assertEqual(len(identify_through_gradient(self.df)[1]), self.hits,
                        msg="Detected %r hits. Expected to detect %r." %\
                        (len(identify_through_gradient(self.df)[1]),self.hits))

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
    def setUp(self):
        func = lambda t: array('d',(max(a,0) for a in np.sin(2*np.pi*t)))
    
        self.time_array = np.linspace(0,1,1000)

        self.samples = func(self.time_array)

        self.a = Abuelmaatti(self.time_array, self.samples)

    def test_abuelmaatti_gamma_function_returns_05(self):
     
        self.assertAlmostEqual(self.a.gamma(1), 0.5, places=3,
                               msg="Calculated value for gamma(1) is 0.5. "\
                                 "Value returned was %r." % self.a.gamma(1))

    def test_abuelmaatti_gamma_function_returns_0(self):
        for i in range(1,11):
            self.assertAlmostEqual(self.a.gamma(2), 0, places=3,
                                   msg="Calculated value for gamma(%r) is 0. "\
                                     "Value returned was %r." \
                                     % (i, self.a.gamma(2)))

    def test_abuelmaatti_delta_function_returns_expected_values(self):
        expected_deltas = [0, -0.212, 0, -0.042, 0, -0.018,
                           0, -0.0098,0, -0.0064]
        delta_0 = 0.318

        self.assertAlmostEqual(self.a.delta_0, delta_0, places=3,
                               msg="Calculated value for delta_0 is %r. Value"\
                               " returned was %r." % (delta_0, self.a.delta_0))
        for i in range(1,11):
            self.assertAlmostEqual(self.a.delta(i), expected_deltas[i-1],
                                   places=3, 
                                   msg="Calculated value for delta(%r) is %r."\
                                   " Value returned was %r." % (i, 
                                   expected_deltas[i-1], self.a.delta(i)))

    def test_abuelmaatti_returns_equal_values_for_regions_of_equal_phase(self):
        func = lambda t: array('d',(max(a,0) for a in np.sin(np.pi*t)))
        samples = func(self.time_array)
        a = Abuelmaatti(self.time_array[:int(len(samples)/2)],
                        samples[:int(len(samples)/2)])
        b = Abuelmaatti(self.time_array[:int(len(samples)/2)], 
                        samples[int(len(samples)/2):])

        for i in range(1,11):
            self.assertAlmostEqual(a.delta(i), b.delta(i), places=3,
                                   msg="%r and %r do not match for the %rth "
                                   "harmonic." % (a.delta(i), b.delta(i), i))

class TestHitDetectorDensityFuncs(unittest.TestCase):
    def setUp(self):
        point_array = np.linspace(1,100,100)
        self.df = pd.DataFrame(data=dict(rate = point_array,
                                    w1_rate = 0))
        expected_density = np.array([10*[x] for x in range(1,101)]).reshape(1,
                                                                   1000)[0]
        self.expected_density = np.insert(expected_density,0,0)
    def test_point_density_returns_expected_height_array(self):   
        try:
            np.testing.assert_array_almost_equal(point_density(self.df)[0], 
                                                 np.arange(0,100.1,0.1),
                                                 decimal=3)
        except(AssertionError):
            raise(AssertionError("%r and %r are not equal." \
                                 % (point_density(self.df)[0],
                                    np.arange(0,100.1,0.1))))
    def tests_point_density_returns_expected_density_array(self):
        try:
            np.testing.assert_array_almost_equal(point_density(self.df)[1],
                                                 self.expected_density,
                                                 decimal=3)
        except(AssertionError):
            raise(AssertionError("%r and %r are not equal."\
                                 % ((point_density(self.df)[1]),
                                    (self.expected_density))))
#------------hitsimulator.py tests---------------------------------------------
class TestHitSimulatorNumericalFuncs(unittest.TestCase):
    def test_hit_distribution_returns_correct_values(self):
        self.assertTrue(hit_distribution(1)[0][0] <= 2*np.pi)
        self.assertTrue(hit_distribution(1)[0][1] <= 4.5)
    

    def test_flux_expected_values_mass(self):
    # Test the flux function returns expected values.
        self.assertAlmostEqual(flux(2.7e-11), 5.388e-6, places=4, 
                               msg="Flux of particles of mass 2.7e-11 " \
                               "%r. Expected %r." % \
                               (flux(2.7e-11), 5.388e-6))

        self.assertAlmostEqual(flux(2.9e-11), 5.114e-6, places=4, 
                               msg="Flux of particles of mass 2.9e-11 " \
                               "returned %r. Expected %r." % \
                               (flux(2.9e-11), 5.114e-6))

    def test_freq_independent_of_mass_array(self):
    # The total frequency of a range should be the same irrespective of 
    # the array of masses passed.
        masses2 = np.linspace(1e-13, 1e-7, 100) 
        masses1_freq = sum(freq(masses))
        masses2_freq = sum(freq(masses2))
        self.assertAlmostEqual(masses1_freq, masses2_freq, places=4, 
                               msg="Total frequency of 100 mass array " \
                               "calculated as %r. Total frequency of " \
                               "10000 mass array calculated as %r." % \
                               (masses2_freq, masses1_freq))

    def test_p_distribution_returns_expected_shape(self):
        dist = p_distribution(np.linspace(0, 1, 1000))
        self.assertLessEqual(len(dist[1]), len(dist[0]),
                             msg="The array of hit indices was not less than"\
                                  " the array of measured points.")



class TestHitSimulatorGeneratorFuncs(unittest.TestCase):
    
    def test_expected_rate_of_impacts(self):
    # Test that generate_event creates expected rate of impact.
    # Conservative estimate used, expected rate is around 1% so it is 
    # run 100 times and tested that the total amount is no greater than 
    # 8. (Probability(X > 8 = 1e-6.)

    # Since this test is probabalistic it is difficult to test for 
    # certain. It is therefore at the user's discretion (and statistical
    # understanding) to decide at what point a failure truly indicates 
    # failure. 
        frequencies = freq(masses)
        count = 0
        for _ in range(1000):
            # int(bool( to turn any non zero number into 1.
            count += 1*int(bool(generate_event(masses, frequencies)[0])) 
        try:
            self.assertGreater(80, count)
        except(AssertionError):
            # Poisson distribution with rate parameter 1 - not exact 
            # but indicative.
            poisson_1000 = lambda x: (np.e**(-10))/math.factorial(x)
            # Probability of receiving a value of count or higher.
            probability = 1 - sum([poisson_100(x) for x in range(count)])

            raise RuntimeError("\n\n***\nUnexpectedly high hitrate produced " \
                               "by generate_data. Consider re-running tests." \
                               "\n1000 attempts yielded %r hits. The " \
                               "probability of this occurring is less than " \
                               "%.2e.\nIf this happens multiple times, use " \
                               "your statistical discretion to consider this "\
                               "a failure.\n***\n\n" % (count, probability))
            

#------------characteristics.py tests------------------------------------------
class TestResponseTurningPointFuncs(unittest.TestCase):

    def setUp(self):
    # Set up dummy polynomial data with known number of turning points.
        obmt = np.linspace(0,100,1000)
        # Random number of turning points between 0 and 20.
        self.points = np.random.randint(1,20) 

        # Set up rate as a sin function over obmt with the expected 
        # number of turning points.
        rate = np.sin(self.points*np.pi*obmt/100) 
        w1_rate = np.zeros(1000)
        self.df = pd.DataFrame(data=dict(obmt = obmt,
                                         rate = rate,
                                         w1_rate = w1_rate))

    def test_get_turning_points(self):
        self.assertEqual(len(get_turning_points(self.df)),self.points)


    def test_filter_turning_points(self):
        self.assertTrue(len(filter_turning_points(self.df)) <= \
                        len(get_turning_points(self.df)))

#-------------------------------------------

if __name__ == "__main__":
    unittest.main()
