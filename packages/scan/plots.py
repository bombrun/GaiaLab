# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#def plot_att(scan):


def plot_attitude(satellite, ti, tf, dt):
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
    n_steps = (tf-ti)/dt
    t = np.linspace(ti, tf, n_steps)

    qt_list = []
    qx_list = []
    qy_list = []
    qz_list = []

    satellite.reset_to_time(ti)
    for i in np.arange(n_steps):
        satellite.update(dt)
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


def plot_longlat(satellite, dt, n):
    
    satellite.reset()
    lambda_list = [] 
    beta_list = [] 
    for i in np.arange(n/dt):
        satellite.update(dt)
        lambda_list.append(satellite.lamb_z%(2*np.pi)) 
        beta_list.append(satellite.beta_z)  
   
    plt.figure()
    plt.plot(np.degrees(lambda_list), np.degrees(beta_list), 'b.')
    plt.ylabel('Lattitude º')
    plt.xlabel('Longitud º')
    plt.ylim(-np.pi/2, np.pi/2)
    
    plt.rcParams.update({'font.size': 22})
    plt.title('Revolving scanning')
    plt.show()


def plot_xi(satellite, dt, n):

    satellite.Reset()
    angle_list = []
    times = np.arange(0, n, dt)
    for i in times:
        satellite.Update(dt)
        ecliptic_quat = rotation_quaternion(np.array([1, 0, 0]), -satellite.epsilon)
        z_ecliptic = quaternion_to_vector(ecliptic_quat*vector_to_quaternion(satellite.z_)*ecliptic_quat.conjugate())
        s_rot_quat = rotation_quaternion(np.array([0, 0, 1]), satellite.l)
        s_ecliptic = quaternion_to_vector(s_rot_quat*Quaternion(0, 1, 0, 0)*s_rot_quat.conjugate())
        angle = np.arccos(np.dot(z_ecliptic, s_ecliptic))
        
        angle_list.append(np.degrees(angle))
    
    percentage = np.abs(angle_list[0]- angle_list[-1])
    plt.figure()
    plt.grid()
    plt.plot(times, angle_list, 'g.')
    plt.ylabel('Angle (deg)')
    plt.xlabel('Time (days)')
    plt.title('Xi Angle, increment = %f (deg)'%percentage)
    plt.show()    
    
                          
def plot_3DX(satellite, dt, n):
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


def plot_3DZ(satellite, dt, n):
        
    satellite.reset()
    z_list = []
    for i in np.arange(n/dt):
        satellite.Update(dt)
        z_list.append(quaternion_to_vector(frame*vector_to_quaternion(satellite.z_)*frame.conjugate()))
    
    z_list_x = [i[0] for i in z_list]
    z_list_y = [i[1] for i in z_list]
    z_list_z = [i[2] for i in z_list]
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(z_list_x, z_list_y, z_list_z,'--', label='Z vector rotation')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    
    plt.show()         


def plot_3DW(satellite, dt, n):

    satellite.reset()
    w_list = []
    for i in np.arange(n/dt):
        satellite.Update(dt)
        w_list.append(satellite.w_)
    
    w_list_x = [i[0] for i in w_list]
    w_list_y = [i[1] for i in w_list]
    w_list_z = [i[2] for i in w_list]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(w_list_x, w_list_y, w_list_z,'--', label='W: inertial rotation vector')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')
    
    plt.show() 


def plot_observations(scanner, sky):

    x = [i[0] for i in scanner.stars_positions]
    y = [i[1] for i in scanner.stars_positions]
    z = [i[2] for i in scanner.stars_positions]
    
    xstar = [i.coor[0] for i in sky.elements]
    ystar = [i.coor[1] for i in sky.elements]
    zstar = [i.coor[2] for i in sky.elements]

    fig, (ax1, ax2,  ax3) = plt.subplots(1, 3)
    fig.subplots_adjust(left=0.2, wspace=0.6)
    
    ax1.plot(x, y,'ro')
    ax1.plot(xstar, ystar, 'b*', ms=10)
    ax1.set(title='XY PLANE')

    ax2.plot(y, z, 'ko')
    ax2.plot(ystar, zstar, 'b*', ms = 10)
    ax2.set(title='YZ PLANE')

    ax3.plot(x, z, 'go')
    ax3.plot(xstar, zstar, 'b*', ms = 10)
    ax3.set(title='XZ PLANE')

    plt.show()


def plot_diff(satellite, sky):
    diff_list = []
    for star in sky.elements:
        for obj in satellite.storinglist:
            diff_ = np.abs(obj[8] - star.coor)
            diff_list.append(mag(diff_))
    plt.figure()
    plt.plot(diff_list, 'bo--')
    plt.show()
    
    

