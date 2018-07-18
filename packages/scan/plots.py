# -*- coding: utf-8 -*-
from NSL import*

from sympy import*
from sympy import Line3D, Point3D
from sympy import Symbol

import numpy as np
import math
import time
import datetime

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm


########################## PLOTS ################################
            


def PlotAttitude(satellite, dt, n0, nf):
    '''
    Args
    ______
    satellite: object to be calculated and updated.
    dt: step per day, i.e fraction of a day.
    n0: initial day 
    nf: final day
    
    Returns
    ________
    Plot of the 4 components of the attitude of the satellite.
    attitude = (t, x, y, z)
    Each graph plots time in days versus each component evolution wrt time.
    '''
    satellite.Reset()
    t = np.linspace(n0, nf, (nf-n0)/dt)
    qt_list = []
    qx_list = []
    qy_list = []
    qz_list = []
    
    satellite.t = n0
    for i in np.arange((nf-n0)/dt): #number of steps
        satellite.Update(dt)
        qt_list.append(satellite.attitude.w)
        qx_list.append(satellite.attitude.x)
        qy_list.append(satellite.attitude.y)
        qz_list.append(satellite.attitude.z)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.subplots_adjust(left=0.2, wspace=0.6)
    
    ax1.plot(t, qt_list,'ro--')
    ax1.set(title='W', xlabel = 'days')

    ax2.plot(t, qx_list, 'bo--')
    ax2.set(title='X', xlabel = 'days')

    ax3.plot(t, qy_list, 'go--')
    ax3.set(title='Y', xlabel = 'days')
    
    ax4.plot(t, qz_list, 'ko--')
    ax4.set(title='Z', xlabel = 'days')
    
    plt.rcParams.update({'font.size': 22})
    
    plt.show()
   
def PlotLatLong(satellite, dt, n):
    
    satellite.Reset()
    lambda_list = [] 
    beta_list = [] 
    for i in np.arange(n/dt):
        satellite.Update(dt)
        lambda_list.append(satellite.lamb_z%(2*np.pi)) 
        beta_list.append(satellite.beta_z)  
   
    plt.figure()
    plt.plot(np.degrees(lambda_list), np.degrees(beta_list), 'b.')
    plt.ylabel('Lattitude º')
    plt.xlabel('Longitud º')
    plt.ylim(-np.pi/2, np.pi/2)
    
    plt.rcParams.update({'font.size': 22})
    #plt.title('Revolving scanning')
    plt.show()
        

    

def PlotXi(satellite, dt, n):

    satellite.Reset()
    angle_list = []
    times = np.arange(0, n, dt)
    for i in times:
        satellite.Update(dt)
        ecliptic_quat = rotation_quaternion(np.array([1,0,0]),-satellite.epsilon)
        z_ecliptic = quaternion_to_vector(ecliptic_quat*vector_to_quaternion(satellite.z_)*ecliptic_quat.conjugate())
        s_rot_quat = rotation_quaternion(np.array([0,0,1]), satellite.l)
        s_ecliptic = quaternion_to_vector(s_rot_quat*Quaternion(0,1,0,0)*s_rot_quat.conjugate())
        angle = np.arccos(np.dot(z_ecliptic,s_ecliptic))
        
        angle_list.append(np.degrees(angle))
    
    percentage = np.abs(angle_list[0]- angle_list[-1])
    plt.figure()
    plt.grid()
    plt.plot(times, angle_list, 'g.')
    plt.ylabel('Angle (deg)')
    plt.xlabel('Time (days)')
    plt.title('Xi Angle, increment = %f (deg)'%percentage)
    plt.show()    
    
                          
def Plot3DX(satellite, dt, n):
    satellite.Reset()
    x_list = []
    for i in np.arange(n/dt):
        satellite.Update(dt)
        x_quat = satellite.attitude * Quaternion(0,1,0,0) * satellite.attitude.conjugate() 
        x_ = quaternion_to_vector(x_quat)
        x_list.append(x_)
    
    x_listx = [i[0] for i in x_list]
    x_listy = [i[1] for i in x_list]
    x_listz = [i[2] for i in x_list]
    
    
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x_listx, x_listy, x_listz,'--', label='X vector rotation')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    plt.show() 
    
def Plot3DZ(satellite, dt, n, frame = None):
    #frame allows a quaternion rotation to be applied to the z_ vector before being plotted, e.g. rotation_quaternion(np.array([1,0,0]),-gaia.epsilon) will move from the lmn frame to the ecliptic plane.
    if frame == None:   #frame allows a quaternion rotation to be applied to the z_ vector before being plotted, e.g. rotation_quaternion(np.array([1,0,0]),-gaia.epsilon) will move from the lmn frame to the ecliptic plane.
        frame = Quaternion(1,0,0,0)
        
    satellite.Reset()
    z_list = []
    for i in np.arange(n/dt):
        satellite.Update(dt)
        z_list.append(quaternion_to_vector(frame*vector_to_quaternion(satellite.z_)*frame.conjugate()))     #z = frame*z_vector*frame.conj
    
    z_listx = [i[0] for i in z_list]
    z_listy = [i[1] for i in z_list]
    z_listz = [i[2] for i in z_list]
    
    
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(z_listx, z_listy, z_listz,'--', label='Z vector rotation')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    
    plt.show()         
     
def Plot3DW(satellite, dt, n):
    satellite.Reset()
    w_list = []
    for i in np.arange(n/dt):
        satellite.Update(dt)
        w_list.append(satellite.w_)
    
    w_listx = [i[0] for i in w_list]
    w_listy = [i[1] for i in w_list]
    w_listz = [i[2] for i in w_list]
    
    
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(w_listx, w_listy, w_listz,'--', label='W: inertial rotation vector')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')
    
    plt.show() 

def PlotStar(scanner, sky):
    scanner.SortPositions()
    x = [i[0] for i in scanner.starspositions]
    y = [i[1] for i in scanner.starspositions]
    z = [i[2] for i in scanner.starspositions]
    xstar = [i.coor[0] for i in sky.elements]
    ystar = [i.coor[1] for i in sky.elements]
    zstar = [i.coor[2] for i in sky.elements]
    
    mpl.rcParams['legend.fontsize'] = 18
    
    # 3D PLOT
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x, y, z,'--', label='Observations')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    
    # 2D PLOT
    fig, ((ax1, ax2,  ax3)) = plt.subplots(1,3)
    fig.subplots_adjust(left=0.2, wspace=0.6)
    
    ax1.plot(x, y,'ro')
    ax1.plot(xstar, ystar, 'b*', ms = 15)
    ax1.set(title='XY PLANE')
    

    ax2.plot(y, z, 'ko')
    ax2.plot(ystar, zstar, 'b*', ms = 15)
    ax2.set(title='YZ PLANE')

    ax3.plot(x, z, 'go')
    ax3.plot(xstar, zstar, 'b*', ms = 15)
    ax3.set(title='XZ PLANE')

    
    plt.rcParams.update({'font.size': 22})
    
    plt.show()
       

    
    

