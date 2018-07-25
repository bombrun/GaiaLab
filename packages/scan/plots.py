# -*- coding: utf-8 -*-


import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from quaternion import Quaternion


def plot_longalt(att, ti, tf, dt):
    att.reset()
    att.create_storage(ti, tf, dt)
    long_list = [i[5]%(2*np.pi) for i in att.storage]
    alt_list = [i[6] for i in att.storage]

    plt.figure()
    plt.plot(np.degrees(long_list), np.degrees(alt_list), 'b.')
    plt.xlabel('Longitud º')
    plt.ylabel('Lattitude º')

    #plt.ylim(-np.pi/2, np.pi/2)
    
    plt.rcParams.update({'font.size': 22})
    plt.title('Revolving scanning')
    plt.show()
                          
def plot_3DX(att, ti, tf, dt):

    att.reset()
    att.create_storage(ti, tf, dt)
    x_list = [obj[3] for obj in att.storage]

    x_listx = [i[0] for i in x_list]
    x_listy = [i[1] for i in x_list]
    x_listz = [i[2] for i in x_list]

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x_listx, x_listy, x_listz, '--', label='X vector rotation')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')

    plt.show()


def plot_3DZ(att, ti, tf, dt):
    att.reset()
    att.create_storage(ti, tf, dt)
    z_list = [obj[2] for obj in att.storage]

    z_listx = [i[0] for i in z_list]
    z_listy = [i[1] for i in z_list]
    z_listz = [i[2] for i in z_list]

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(z_listx, z_listy, z_listz, '--', label='Z vector rotation')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')

    plt.show()

def plot_3DW(att, ti, tf, dt):
    att.reset()
    att.create_storage(ti, tf, dt)
    w_list = [obj[1] for obj in att.storage]

    w_listx = [i[0] for i in w_list]
    w_listy = [i[1] for i in w_list]
    w_listz = [i[2] for i in w_list]

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(w_listx, w_listy, w_listz, '--', label='W vector rotation')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')

    plt.show()

def plot_3Dsun(att, ti, tf, dt):
    att.reset()
    att.create_storage(ti, tf, dt)
    sun_list = [obj[7] for obj in att.storage]
    x_srs_ecliptical = [i[0] for i in sun_list]
    y_srs_ecliptical = [i[1] for i in sun_list]
    z_srs_ecliptical = [i[2] for i in sun_list]

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x_srs_ecliptical, y_srs_ecliptical, z_srs_ecliptical, 'bo--', label='s vector /sun movement wrt SRS-ecliptical')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')

    plt.show()


def plot_attitude(att, ti, tf, dt):
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

    att.reset()
    att.create_storage(ti, tf, dt)
    attitude_list = [obj[4] for obj in att.storage]

    qw_list = [obj.w for obj in attitude_list]
    qx_list = [obj.x for obj in attitude_list]
    qy_list = [obj.y for obj in attitude_list]
    qz_list = [obj.z for obj in attitude_list]
    times = [obj[0] for obj in att.storage]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.subplots_adjust(left=0.2, wspace=0.6)

    ax1.plot(times, qw_list, 'ro--')
    ax1.set(title='W', xlabel='days')

    ax2.plot(times, qx_list, 'bo--')
    ax2.set(title='X', xlabel='days')

    ax3.plot(times, qy_list, 'go--')
    ax3.set(title='Y', xlabel='days')

    ax4.plot(times, qz_list, 'ko--')
    ax4.set(title='Z', xlabel='days')

    plt.rcParams.update({'font.size': 22})

    plt.show()

def plot_observations(scanner, sky):

    x = [i[0] for i in scanner.stars_positions]
    y = [i[1] for i in scanner.stars_positions]
    z = [i[2] for i in scanner.stars_positions]
    
    x_star = [i.coor[0] for i in sky.elements]
    y_star = [i.coor[1] for i in sky.elements]
    z_star = [i.coor[2] for i in sky.elements]

    fig1, (ax1, ax2,  ax3) = plt.subplots(1, 3)
    fig1.subplots_adjust(left=0.2, wspace=0.6)
    
    ax1.plot(x, y,'ro')
    ax1.plot(x_star, y_star, 'b*', ms=10)
    ax1.set(title='XY PLANE')

    ax2.plot(y, z, 'ko')
    ax2.plot(y_star, z_star, 'b*', ms = 10)
    ax2.set(title='YZ PLANE')

    ax3.plot(x, z, 'go')
    ax3.plot(xstar, zstar, 'b*', ms = 10)
    ax3.set(title='XZ PLANE')


    mpl.rcParams['legend.fontsize'] = 10

    fig2 = plt.figure()
    ax = fig2.gca(projection='3d')

    ax.plot(x, y, z, '--', label='observation measurements')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

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
    
    

