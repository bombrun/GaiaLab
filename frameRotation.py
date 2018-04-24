"""
Alex Bombrun
Simple frame rotation comparison between 2 cataloques
from LL-072 Equations (36) and (37) without galactic acceleration
"""

import numpy as np
import time
import matplotlib as mp
import matplotlib.pyplot as plt
import pandas as pd
import healpy as hp
import glob as glob

from astropy.table import Table

identity = np.ones(3)/np.linalg.norm(np.ones(3))

def alphaDelta(r) :
    """
    from the direction r of the frame pqr computes and return 
    (alpha,delta) : right ascention, declination in radian
    """
    x,y,z = tuple(r)
    cosDelta = np.sqrt(x**2+y**2)
    delta = np.arctan2(z,cosDelta)
    alpha = np.arctan2(y/cosDelta,x/cosDelta)
    return alpha%(2*np.pi),delta

def normalised(r):
    x,y,z = tuple(r)
    return (x,y,z)/np.sqrt(x**2,y**2,z**2)    
    

def pqr(alpha,delta) :
    """
    comptutes and returns the frame pqr for a source at (alpha,delta) 
    alpha : right ascension in radian
    delta : declination in radian 
    """
    p=np.array([-np.sin(alpha),np.cos(alpha),0])
    q=np.array([-np.sin(delta)*np.cos(alpha),-np.sin(delta)*np.sin(alpha),np.cos(delta)])
    r=np.array([np.cos(delta)*np.cos(alpha),np.cos(delta)*np.sin(alpha),np.sin(delta)])
    return p,q,r

def qOmega(omega,t0,t1):
    """
    Equation 5 in LL-072
    
    """
    r = np.linalg.norm(omega)
    if r>0.0 :
        v = omega/r*np.sin((t0-t1)*r/2)
    else :
        v = np.zeros(3)
    return v[0],v[1],v[2],np.cos((t0-t1)*r/2)

def qEpsilon(epsilon):
    """
    Equation 8 in LL-072
    
    """
    r = np.linalg.norm(epsilon)
    if r>0.0 :
        v = epsilon/r*np.sin(r/2)
    else :
        v = np.zeros(3)
    return v[0],v[1],v[2],np.cos(r/2)


def q_star(q) :
    """
    quaternion conjugate
    """
    x,y,z,w = q
    return -x,-y,-z,w

def q_mult(qa, qb):
    """
    quaternion multiplication
    """
    ax, ay, az, aw = qa
    bx, by, bz, bw = qb
   
    x = ax*bw + ay*bz - az*by + aw*bx
    y = -ax*bz + ay*bw + ax*bx + aw*by
    z = ax*by - ay*bx + az*bw + aw*bz
    w = -ax*bx - ay*by - az*bz + aw*bw

    return x, y, z, w

def q_norm(q):
    x,y,z,w = q
    return np.sqrt(x**2+y**2+z**2+w**2)

def rotate(q,v):
    qv = np.array([v[0],v[1],v[2],0])
    q_inv = q_star(q)
    rv = q_mult(q,q_mult(qv,q_inv))
    return np.array([rv[0].rv[1],rv[2]])

def EtG(omega,epsilon,t0,t1):
    """
    matrix transformation from frame G to E
    Equation 8 in LL-072
    
    """
    qO = qOmega(omega,t0,t1)
    qE = qEpsilon(epsilon)
    x,y,z,w = q_mult(qO,qE)
    
    return np.array([[w**2+x**2-y**2-z**2, 2*(x*y+z*w), 2*(x*z-y*w)],
            [2*(x*y-z*w), w**2-x**2+y**2-z**2, 2*(y*z+x*w)],
            [2*(x*z+y*w), 2*(y*z-x*w), w**2-x**2-y**2+z**2]
          ])

def r0Emu0E(omega,epsilon,t0,t1,alphaG,deltaG,muAlphaStarG,muDeltaG) :
    """
    computes and returns the direction and the propermotion vectors in the reference frame defined by omega, epsilon, t1
    of a source with position (alphaG,deltaG) at t0 and (muAlphaStarG,muDeltaG) 
    
    Equation 21 and 25 in LL-072
    """
    p0G,q0G,r0G = pqr(alphaG,deltaG)
    
    muG = muAlphaStarG * p0G + muDeltaG * q0G - np.cross(omega,r0G)
    
    EtG0 = EtG(omega,epsilon,t0,t1)
    
    r0E = EtG0.dot(r0G)
    
    mu0E = EtG0.dot(muG)
    
    return r0E, mu0E


def designEquationRow(omega,epsilon,
                      t0,t1,
                      alphaG,deltaG,muAlphaStarG,muDeltaG,
                      alpha0E,delta0E,muAlphaStar0E,muDelta0E) :
    """
    Equation 34,35 in LL-072
    """
    
    
    r0EG, mu0EG = r0Emu0E(omega,epsilon,t0,t1,alphaG,deltaG,muAlphaStarG,muDeltaG)
    alpha0EG, delta0EG = alphaDelta(r0EG)
    p0E,q0E,r0E = pqr(alpha0EG, delta0EG)
    
    p2E,q2E,r2E = pqr(alpha0E, delta0E)
    
    rAlpha = p0E.dot(r0E-r2E)
    rDelta = q0E.dot(r0E-r2E)
    rMuAlpha = p0E.dot(mu0EG)-muAlphaStar0E
    rMuDelta = q0E.dot(mu0EG)-muDelta0E

    z = np.zeros(3)
    
    drAlpha = np.array([-q0E,-q0E*(t0-t1)]).flatten()
    drDelta = np.array([p0E,p0E*(t0-t1)]).flatten()
    drMuAlpha = np.array([z,-q0E]).flatten()
    drMuDelta = np.array([z,p0E]).flatten()
    
    return rAlpha,rDelta, rMuAlpha , rMuDelta, drAlpha, drDelta, drMuAlpha, drMuDelta

def designEquation(omega,epsilon,
                   a00,d00,ma00,md00,t0,
                   a11,d11,ma11,md11,t1) :
    """
    00 : Gaia source catalogue 
    t0 : refence epoch of 00
    11 : reference source catalogue
    t1 : reference epoch of 11
    """
    n=len(a00)
    M = np.zeros([4*n,6])
    r = np.zeros(4*n)
    
    for i,a0,d0,ma0,md0,a1,d1,ma1,md1 in zip(range(n),a00,d00,ma00,md00,a11,d11,ma11,md11) :
                
        Ra,Rd,Rma,Rmd,dRa,dRd,dRma,dRmd = designEquationRow(omega,epsilon,t0,t1,a0,d0,ma0,md0,a1,d1,ma1,md1)
        
        M[2*i] = dRa
        M[2*i+1]= dRd
        M[2*i+2] = dRma
        M[2*i+3]= dRmd
        
        r[2*i] = Ra
        r[2*i+1] = Rd
        r[2*i+2] = Rma
        r[2*i+3] = Rmd
        
    return M,r

def solveRotation(i,d,omega=np.zeros(3),epsilon=np.zeros(3)) :
    for i in range(0,i) :
        M,r = designEquation(omega,epsilon,d.alpha,d.delta,d.muAlphaStar,d.muDelta,2015.5,
                             d.alpha0,d.delta0,d.muAlphaStar0,d.muDelta0,2015.5)
        x = np.linalg.lstsq(M,r)
        omega -= x[0][3:6]
        epsilon -=  x[0][0:3]
    return epsilon, omega 

def rotationPerMagPd(df0,df1,gmag0,gmag1,nmax=1000) :
    """
    a pandas implementation
    the magnitude filtering is done in spark, 
    the filtered data is exported to pandas,
    the processing is done locally
    """
    d0=df0.filter((df0.gMag>=gmag0) & (df0.gMag<gmag1)).toPandas()
    d1=df1.filter((df1.gMag>=gmag0) & (df1.gMag<gmag1)).toPandas()
    d0.index=d0.sourceId
    d1.index=d1.sourceId
    d0.sort_index(inplace=True)
    d1.sort_index(inplace=True)
    
    d=d1
    if 'alpha0' not in d.columns :
        d['alpha0']=d0.alpha
    d['delta0']=d0.delta
    d['muAlphaStar0']=d0.muAlphaStar
    d['muDelta0']=d0.muDelta
    
    n=d.alpha.count()
   
    if(n>nmax) : d=d.sample(frac=nmax*1.0/n)
    print(n,": processing",d.alpha.count(),"sources")
    return solveRotation(2,d)
    
def rotationPerColorPd(df0,df1,nuEff0,nuEff1,nmax=1000) :
    """
    a pandas implementation
    the color filtering is done in spark, 
    the filtered data is exported to pandas,
    the processing is done locally
    """
    d0=df0.filter((df0.nuEff>=nuEff0) & (df0.nuEff<nuEff1)).toPandas()
    d1=df1.filter((df1.nuEff>=nuEff0) & (df1.nuEff<nuEff1)).toPandas()
    d0.index=d0.sourceId
    d1.index=d1.sourceId
    d0.sort_index(inplace=True)
    d1.sort_index(inplace=True)
    
    d=d1
    d['alpha0']=d0.alpha
    d['delta0']=d0.delta
    d['muAlphaStar0']=d0.muAlphaStar
    d['muDelta0']=d0.muDelta
    
    n=d.alpha.count()
   
    if(n>nmax) : d=d.sample(frac=nmax*1.0/n)
    print(n,": processing",d.alpha.count(),"sources")
    return solveRotation(2,d)

def rotationPerMag(df0,df1,gmag0,gmag1,nmax=1000) :
    """
    a spark implementation 
    not really faster than the pandas one, except maybe if there is a large number of sources
    """
    d0=df0.filter((df0.gMag>=gmag0) & (df0.gMag<gmag1))
    d1=df1.filter((df1.gMag>=gmag0) & (df1.gMag<gmag1))
    # rename columns in d1
    # d0 = d0.select(*(col(x).alias(x + '_0') for x in d0.columns))
    d1 = d1.select(*(col(x).alias(x + '1') for x in d1.columns))
    # join
    d=d0.join(d1,d0.sourceId == d1.sourceId1,"outer")
    
    n=d.count()
    if(n>nmax) : d=d.sample(False,nmax*1.0/n,1234)
    
    df=d.toPandas()
    df['alpha0']=df.alpha
    df['delta0']=df.delta
    df['muAlphaStar0']=df.muAlphaStar
    df['muDelta0']=df.muDelta
    print(n,": processing",df.alpha.count(),"sources")
    return solveRotation(2,df)


def rotationPerColor(df0,df1,nuEff0,nuEff1,nmax=1000) :
    """
    a spark implementation 
    not really faster than the pandas one, except maybe if there is a large number of sources
    """
    d0=df0.filter((df0.nuEff>=nuEff0) & (df0.nuEff<nuEff1))
    d1=df1.filter((df1.nuEff>=nuEff0) & (df1.nuEff<nuEff1))
    # rename columns in d1
    # d0 = d0.select(*(col(x).alias(x + '_0') for x in d0.columns))
    d1 = d1.select(*(col(x).alias(x + '1') for x in d1.columns))
    # join
    d=d0.join(d1,d0.sourceId == d1.sourceId1,"outer")
    
    n=d.count()
    if(n>nmax) : d=d.sample(False,nmax*1.0/n,1234)
    
    df=d.toPandas()
    df['alpha0']=df.alpha
    df['delta0']=df.delta
    df['muAlphaStar0']=df.muAlphaStar
    df['muDelta0']=df.muDelta
    print(n,": processing",df.alpha.count(),"sources")
    return solveRotation(2,df)