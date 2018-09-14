"""
Hit simulation of micrometeoroids on Gaia

Based on Lennart Lindegren's [SAG--LL-030 technical note]
(http://www.astro.lu.se/~lennart/Astrometry/TN/Gaia-LL-031-20000713-
Effects-of-micrometeoroids-on-GAIA-attitude.pdf).

Importable hit simulation functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import configparser
from numba import jit  # Compiles python - speeds up iteration.

from .misc import s2o, o2s  # Conversion functions for obmt and seconds.

# -----------------------------------------------------------------------------

#
# Functions and classes     test implemented?
#
# hit_distribution          yes
# flux                      yes
# p_distribution            yes
# freq                      yes
# tp_distribution           yes
# time_distribution         yes
# AOCSResponse              yes
# generate_event            yes
# generate_data             yes
#
# -----------------------------------------------------------------------------

# Use hits.ini to decide which variables to use--------------------------------
config = configparser.ConfigParser()
config.read(__file__[:-15] + 'hits.ini')
_use_defaults = False
# This is really hacky but it checks that the keys are in their
# respective dicts before using them to avoid throwing an exception.
if 'hitsimulator' in config and 'use_conf' in config['hitsimulator'] and \
                            config.getboolean('hitsimulator', 'use_conf'):
    try:
        from .conf import r, v, I, R, masses, tp_m_loc, tp_m_scale, tp_c_loc, \
                            tp_c_scale, t_m_loc, t_m_scale, t_c_loc, t_c_scale
    except(ImportError):
        _use_defaults = True
else:
    _use_defaults = True

if _use_defaults:
    r = 3     # m         typical impact distance from z axis
    v = 12e3  # m/s       rms tangential velocity of particle
    I = 7e3   # kg m^2    spacecraft moment of inertia about z axis
    R = 4.25  # m         spacecraft radius

    # Define the mass spectrum to be used. More masses => greater
    # accuracy but doesn't affect hit rate. 10000 is the default but
    # different sized arrays can be used.

    masses = np.linspace(1e-13, 1e-7, 10000)  # kg

    # Only masses between 1e-13 and 1e-7 need be considered - lower than
    # 1e-13 have undetectable impacts, higher than 1e-7 have vanishingly
    # small hit probabilities

    # Define the scale and loc for the time and turning point
    # distributions. These are estimates not calculated from data. Real
    # values are included in conf.py.
    tp_m_loc = 0.2
    tp_m_scale = 0.05
    tp_c_loc = -1
    tp_c_scale = 0.1

    t_m_loc = 0
    t_m_scale = 0.001
    t_c_loc = 0.004
    t_c_scale = 0.0001

# Function definitions.
# First functions use @jit decorators for speed.


@jit
def hit_distribution(hits):
    """
    Accepts:

        the number of hits to simulate on the satellite.

    A sampler for hits distributed uniformly across a disk.

    Returns:

        a tuple of the angle and the radius of the hits' locations.
    """
    theta = np.random.uniform(0, 2 * np.pi, hits)
    radius = np.sqrt(np.random.uniform(0, R ** 2, hits))
    return [(t, r) for t, r in zip(theta, radius)]


@jit
def flux(mass):  # Typical flux of micrometeoroids with  mass > mass.
    """
    Accepts:

        a mass

    Returns:

        the flux of that mass as predicted by Yamakoshi
        (Extraterrestrial dust, ASSL 181,1994).
    """
    if mass < 2.8e-11:
        return 2.8e-11 * mass ** (-0.5)
    else:
        return 2.6e-18 * mass ** (-7.0/6.0)


@jit
def sigma_o(T):
    """Error on hits of period T as predicted by Lindegren."""
    return 126 * T ** (-1.5)


@jit
def p_distribution(frequencies):
    """
    Accepts:

        an array of frequencies of particle impacts.

    Applies a random sampler from the poisson distribution with each
    frequency as the rate parameter to generate hits.

    Returns:

        a tuple of:

            an array of the number of hits per frequency.

            an array of the total hits.
    """

    hit_dist = [np.random.poisson(lam=max(f, 0)) for f in frequencies]
    # max filters out negative frequencies at the flux discontinuity.

    # Indices of non-zero elements of the hit distribution.
    hits = [i for i, e in enumerate(hit_dist) if e != 0]
    return (hit_dist, hits)


@jit
def freq(masses):
    """
    Accepts:

        an increasing array of masses.

    Applies flux() to each mass and subtracts the flux of the mass
    immediately after. This effectively bins the fluxes.

    Returns:

        an array of frequencies corresponding to the masses given.
    """
    return [100*(flux(m) - flux(m + dm)) for m, dm in zip(masses[:-1],
                                                          np.diff(masses))]


@jit
def tp_distribution(amplitude):
    """
    Accepts:

        the amplitude of a hit.

    Returns the number of turning points in the attitude response.

    Through analysing known hits, the amount of turning points in the
    response were determined as a function of hit amplitude. Assuming
    the fitting parameters to follow a normal distribution with their
    calculated values as the means and their error as the standard
    deviation, a value for the number of turning points in the response
    to a hit of given amplitude can be returned.

    The function returns a float as although only an integer can be used
    to characterise the number of turning points, the extra information
    in a float characterises the magnitude of the "wobbles" in response.

    Returns:

        a number close to the number of expected turning points in the
        hit response.
    """
    m = np.random.normal(loc=tp_m_loc, scale=tp_m_scale)
    c = np.random.normal(loc=tp_c_loc, scale=tp_c_scale)

    return abs(m * amplitude + c + 2)
    # +2 accounts for the initial peak, meaning there is only 1 peak and
    # nothing in the response. This stops responses with negative
    # amounts of turning points being allowed.


@jit
def time_distribution(amplitude):
    """
    Accepts:

        the amplitude of a hit.

    Returns the response time of the spacecraft to a hit of given
    amplitude.

    Through analysing known hits, the response time of the spacecraft
    was determined as a function of hit amplitude. Assuming the fitting
    parameters to follow a normal distribution with their calculated
    values as the means and their error as the standard deviation, a
    value for the response time for a hit of given amplitude can be
    returned.

    Returns:

        the expected response time for a hit.
    """
    m = np.random.normal(loc=t_m_loc, scale=t_m_scale)
    c = np.random.normal(loc=t_c_loc, scale=t_c_scale)

    return abs(m * amplitude + c)


class AOCSResponse:
    """
    A class to produce and hold the expected decay pattern of a hit from
    the amplitude.
    """
# Dunder methods---------------------------------------------------------------
    def __init__(self):
        self._data = []

    def __call__(self, amplitude):
        # Compares calculated display pattern to the current data, adds
        # them elementwise, adding zeros for the places in the shorter
        # array not occupied.
        data = self._decay_pattern(amplitude)
        diff = -(len(self._data) - len(data))
        old = self._data
        self._data = [a + b for a, b in zip(old + diff * [0],
                                            data + (-1) * diff * [0])]

    def __getitem__(self, index):
        # self._data will realistically never exceed 200 so holding it
        # in memory is not an issue.

        # Returns a value and deletes that value.
        return self._data.pop(index)

# Private methods--------------------------------------------------------------
    @staticmethod
    def _decay_pattern(amplitude):
        tps = tp_distribution(amplitude)
        time = time_distribution(amplitude)

        d_omegas = [amplitude * (np.e ** (-2 * t/(o2s(time)))) *
                    np.cos((2 * int(tps + 1) + 1) *
                    t * np.pi/(2 * o2s(time)))
                    for t in range(int(o2s(time)))]

        return d_omegas
# -----------------------------------------------------------------------------

# Two parent functions for generating data sets--------------------------------


def generate_event(masses, frequencies, sigma=False):
    """
    Accepts:

        an array of masses, an array of frequencies.

    Generates events at each second based on the probabilities of impact
    relative to the flux of particles. Returns magnitude of the
    associated displacement in angular velocity and theoretical
    resolution of the event based on the frequency of occurrance. Thus,
    higher mass particles return lower error.

    Since this is called multiple times by generateData(), it is more
    efficient to pass frequencies to this function than to calculate
    them from the masses each time the function is called.

    A default mass range is packaged with these functions (masses).
    It is a linear range of 10,000 masses between 1e-13 and 1e-7 kg.

    The size of the mass array does not affect the hit rate, but rather
    the accuracy to which the hits can be simulated. Recommended formats
    are np.linspace or np.logspace. Recommended mass scales are between
    1e-13 and 1e-7 kg.

    Kwargs:

        sigma (bool, default=False):
            if True, calculates the error on the detected magnitude of
            the hits as predicted by Lindegren.

    Returns:

        a tuple of the change in angular velocity created and the error
        on this change. Will return (0,0) most of the time since hits
        only occur ~1% of the time.
    """
    distribution, hits = p_distribution(frequencies)

    if sigma:
        # Uncertainty in omega for a given period.
        sigma_omega = np.sqrt(sum([(sigma_o(f**(-1)))**2
                                  for f in np.array(frequencies)[hits]])) * \
                                  1e-3  # mas
    else:
        sigma_omega = 0

    d_omega = sum([mass * hit_distribution(1)[0][1]*v/I*(180/np.pi*3600e3)
                  for mass in masses[hits]])  # mas

    if d_omega:
        return (d_omega, sigma_omega)
    else:
        return (0, 0)


def generate_data(length, masses=masses, sigma=False, noise=('gaussian',
                                                             'periodic')):
    """
    Accepts:

        the length of time (in s) to be simulated.

    Kwargs:

        masses (array, default=np.linspace(1e-13,1e-7,10000)):
            Masses to be used to calculate the flux of particles.

            The size of the mass array does not affect the hit rate, but
            rather the accuracy to which the hits can be simulated.
            Recommended formats are np.linspace or np.logspace.

            Recommended mass scales are between 1e-13 and 1e-7 kg.

            Logarithmic mass data leads to more precision for lower mass
            particles, which can be beneficial since they make up the
            majority of hits.

        sigma (bool, default=True):
            passes this to generate_event.

    Returns:

        a Pandas dataframe of shape:

                obmt    rate    error   w1_rate
            1.  float   float   float   float
    """

    frequencies = freq(masses)
    obmt = np.arange(0, length, 1)
    starts = [False]
    sigmas = [0]
    omega = [0]
    response = AOCSResponse()

    for t in range(length-1):
        _omega = generate_event(masses, frequencies, sigma=sigma)
        if _omega[0] != 0:
            response(_omega[0])
            d_omega = _omega[0] + response[0]
            start = True
        else:
            try:
                d_omega = 0 + response[0]
            except(IndexError):  # response._data is an empty list.
                d_omega = 0
            finally:
                start = False
        omega.append(d_omega)
        sigmas.append(_omega[1])
        starts.append(start)

    df = pd.DataFrame({"obmt": s2o(obmt),
                       "rate": omega,
                       "error": sigmas})
    df = df[['obmt', 'rate', 'error']]

    if hasattr(noise, '__iter__') and 'gaussian' in noise:
        gaussian_noise = np.random.normal(0, 0.001, len(df['rate']))
    else:
        gaussian_noise = np.zeros(len(df['rate']))
    if hasattr(noise, '__iter__') and 'periodic' in noise:
        periodic_noise_amplitudes = [x ** 2 for x in np.random.normal(0, 0.2,
                                                                      500)]
        periods = [2 * x for x in range(len(periodic_noise_amplitudes))]
        harmonics = [np.sin(df['obmt'] * abs(x) / 2 * np.pi) for x in
                     periods]

        periodic_noise = sum([x * A for x,
                              A in zip(harmonics,
                                       periodic_noise_amplitudes)])
    else:
        periodic_noise = np.zeros(len(df['rate']))

    df['rate'] = df['rate'] + gaussian_noise + periodic_noise
    df['w1_rate'] = df['rate'].rolling(window=3600, min_periods=0).mean()
    return df
