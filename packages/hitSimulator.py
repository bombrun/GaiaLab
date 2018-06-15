# coding: utf-8

# # Hit simulation of micrometeorites on Gaia
# 
# Based on Lennart Lindegren's [SAG--LL-030 technical note](http://www.astro.lu.se/~lennart/Astrometry/TN/Gaia-LL-031-20000713-Effects-of-micrometeoroids-on-GAIA-attitude.pdf).
#
# Importable hit simulation functions
#
# Updated version uses numba's @jit decorators to speed up iterative processes.


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit #compiles python - speeds up iteration

#values

r = 3 #m            typical impact distance from z axis - use hitDist instead
v = 12e3 #m/s       rms tangential velocity of particle
I = 7e3 #kg m^2     spacecraft moment of inertia about z axis
R = 4.25 #m         spacecraft radius

#Define the mass spectrum to be used. More masses => greater accuracy but doesn't affect hit rate. 10000 is the default but different sized arrays can be used.
masses = np.linspace(1e-13,1e-9,10000) #kg - only masses between e-13 and e-9 need be considered - lower than e-13 have undetectable impacts, higher than e-9 have vanishingly small hit probabilities

#Function definitions
#First functions use @jit decorators for speed.

@jit
def hitDist(hits): #distribution of micrometeorite hits across Gaia - assumed to be uniform across a disk
    theta = np.random.uniform(0,2*np.pi,hits)
    radius = np.sqrt(np.random.uniform(0,R**2,hits))
    return [(t,r) for t, r in zip(theta,radius)]

@jit
def flux(mass):    #typical flux of micrometeorites greater than mass = mass
    if mass < 2.8e-11:
        return 2.8e-11 * mass ** (-0.5)
    else:
        return 2.6e-18 * mass ** (-7.0/6.0)

@jit
def p_distribution(frequencies):
    hit_distribution = [np.random.poisson(lam=max(frequency,0)) for frequency in frequencies] #max filters out negative frequencies at the flux discontinuity
    hits = [i for i, e in enumerate(hit_distribution) if e != 0]
    return (hit_distribution, hits)


@jit
def freq(masses):
    return [100*(flux(m) - flux(m + dm)) for m, dm in zip(masses[:-1], np.diff(masses))] #per second - subtraction of the higher flux effectively bins particles

#Two master functions for generating data sets

def generateEvent(masses, frequencies):#create impacts according to flux for a given mass spectrum and frequency distribution

    """
    Generates events at each second based on the probabilities of impact relative to the flux of particles.
    Returns magnitude of the associated displacement in angular velocity and theoretical resolution
    of the event based on the frequency of occurrance. Thus, higher mass particles return lower error.

    Will return (0,0) most of the time since hits only occur ~1% of the time.
    """
    
    distribution,hits = p_distribution(frequencies)

    sigma_o = lambda T: 126* T**(-1.5) #uncertainty in omega for a given period, determines if hits are detectable
        
    sigma_omega = np.sqrt(sum([(sigma_o(frequency**(-1)))**2 for frequency in np.array(frequencies)[hits]])) * 1e-3  #convert from micro arcseconds to milli arcseconds
    d_omega = sum([mass * hitDist(1)[0][1] * v / I * ( 180/np.pi * 3600e3 ) for mass in masses[hits]]) #converts from radians to milli arcseconds 
    
    if d_omega:
        return (d_omega,sigma_omega)
    else:
        return (0,0)

def generateData(masses, length, plot=False):   #return a pandas dataframe for given masses of a given length of time
    
    """
    Accepts input of an array of masses. The size of the array does not affect the hit rate, but rather 
    the accuracy to which the hits can be simulated. Recommended formats are np.linspace or np.logspace. 
    Recommended mass scales are between e-13 and e-9 kg.

    Logarithmic mass data leads to more precision for lower mass particles, which can be beneficial 
    since they make up the majority of hits.

    'length' is simply the length in seconds for which the simulation should run.

    'plot' returns an errorbar plot of the hits and their associated uncertainty.
    """

    frequencies = freq(masses)
    obmt = np.arange(0, length,1)
    
    sigmas = [0]
    omega = [0]

    for t in range(length-1):
        _omega = generateEvent(masses, frequencies) #okay to pass entire mass array to generateEvent even though the last mass is neglected
        omega.append(_omega[0])
        sigmas.append(_omega[1])

    df = pd.DataFrame({"OBMT" : obmt,
                       "Omega" : omega,
                       "Error" : sigmas})
    df = df[['OBMT','Omega','Error']]

    if plot: #plot argument for ipython interaction. only returns non-zero elements of the dataframe, and plots an errorbar graph of hits
        plt.errorbar(obmt, omega, sigmas, fmt='k.', markersize=0.001, capsize=2)
        plt.xlabel("OBMT/seconds")
        plt.ylabel("Angular velocity/mas/s")
        plt.show()
        return  df[df.Omega!=0]
    else:
        return df
