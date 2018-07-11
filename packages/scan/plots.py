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


########################## PLOTS ################################
            
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

def PlotAttitude(satellite, dt, n):
    '''
    Args
    ______
    satellite: object to be calculated and updated.
    dt: step per day, i.e fraction of a day.
    n: number of days
    
    Returns
    ________
    Plot of the 4 components of the attitude of the satellite.
    attitude = (t, x, y, z)
    Each graph plots time in days versus each component evolution wrt time.
    '''
    satellite.Reset()
    t = np.arange(0, n, dt)
    qt_list = []
    qx_list = []
    qy_list = []
    qz_list = []
    for i in np.arange(n/dt):
        satellite.Update(dt)
        qt_list.append(satellite.attitude.w)
        qx_list.append(satellite.attitude.x)
        qy_list.append(satellite.attitude.y)
        qz_list.append(satellite.attitude.z)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.subplots_adjust(left=0.2, wspace=0.6)
    
    ax1.plot(t, qt_list,'ro--')
    ax1.set(title='Attitude components wrt time', ylabel='qw')

    ax2.plot(t, qx_list, 'bo--')
    ax2.set_ylabel('qx')

    ax3.plot(t, qy_list, 'go--')
    ax3.set_ylabel('qy')
    
    ax4.plot(t, qz_list, 'ko--')
    ax4.set_ylabel('qz')
    
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
    plt.plot(lambda_list, beta_list, 'b.')
    plt.ylabel('Beta_z (rad)')
    plt.xlabel('Lambda_z (rad)')
    plt.ylim(-np.pi/2, np.pi/2)
    plt.title('Revolving scanning')
    plt.show()
        
def PlotLatLongEcliptic(satellite, dt, n):
    # Rotates z_ into ecliptic plane, then applies transformation from cartesian to spherical polar coordinates to extract lat. long. data.
    satellite.Reset()
    lat_list = [] #theta
    long_list = [] #phi
    for i in range(n):
        satellite.Update(dt)
        ecliptic_quat = rotation_quaternion(np.array([1,0,0]), -satellite.epsilon) #change to ecliptic plane
        z_ecliptic = quaternion_to_vector(ecliptic_quat*vector_to_quaternion(satellite.z_)*ecliptic_quat.conjugate()) #z vector wrt ecliptic plane
        long_list.append(np.arctan2(z_ecliptic[1],z_ecliptic[0])) #alpha
        lat_list.append(np.arctan2(z_ecliptic[2], np.sqrt(z_ecliptic[0]**2 + z_ecliptic[1]**2))) #delta
        
    plt.figure()
    plt.plot(long_list, lat_list, 'b.')
    plt.ylabel('Ecliptic Lattitude (rad)')
    plt.xlabel('Ecliptic Longitude (rad)')
    plt.ylim(-np.pi/2, np.pi/2)
    plt.title('Revolving scanning')
    plt.show()
    

def PlotXi(satellite, dt, n):
    #from the PlotLatLong2 function, it looks like this angle is increasing over time... 
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
    plt.plot(times, angle_list, 'b.')
    plt.ylabel('Angle (deg)')
    plt.xlabel('Time (days)')
    plt.title('Xi Angle, increment = %f (deg)'%percentage)
    plt.show()    

def Angle(scanner):
    angles = [i[1] for i in scanner.observations]
    plt.figure()
    plt.grid()
    plt.plot(angles, 'bo')
    plt.ylabel('rad')
    plt.title('Angle between Scanner Axis and star position')
    plt.show()
    
    
def PlotAxis(scanner):
    
    x = [i[0] for i in scanner.axis]
    y = [i[1] for i in scanner.axis]
    z = [i[2] for i in scanner.axis]
    
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x, y, z,'bo', label='x-axis')

    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    
    plt.show()      
    
def PlotObs(scanner,satellite):
    stars_list = []
    for t in scanner.times_obs:
        star_ = satellite.GetXAxis(t)
        stars_list.append(star_)
        
    x = [i[0] for i in stars_list]
    y = [i[1] for i in stars_list]
    z = [i[2] for i in stars_list]
        

    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x, y, z,'b*', label='Measurement')

    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    
    plt.show()         
    
                          
def Plot3DX(satellite, dt, n):
    satellite.Reset()
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
  
