# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 15:39:19 2018

@author: mdelvallevaro
"""

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
import gaia_geometric_toymodel as ggs


def plot_3DX(att, ti, tf, dt):
    """
    %run: plot_3DX(att, 0, 365*5, 0.1)

    :param att: attitude object
    :param ti: initial time [days]
    :param tf: final time [days]
    :param dt: step time for calculating the data point [days]
    :return: plot of the position of the x-axis (unitary) of the scanner wrt LMN frame.
    """
    if isinstance(att, ggs.Attitude) is False:
        raise TypeError('att is not an Attitude object.')
    if type(ti) not in [int, float]:
        raise TypeError('ti must be non-negative real numbers.')
    if type(tf) not in [int, float]:
        raise TypeError('tf must be non-negative real numbers.')
    if type(dt) not in [int, float]:
        raise TypeError('dt must be non-negative real numbers.')
    if ti < 0:
        raise ValueError('ti cannot be negative.')
    if tf <0:
        raise ValueError('tf cannot be negative.')
    if dt <0:
        raise ValueError('dt cannot be negative.')

    att.reset()
    att.create_storage(ti, tf, dt)
    x_list = [obj[3] for obj in att.storage]

    x_listx = [i[0] for i in x_list]
    x_listy = [i[1] for i in x_list]
    x_listz = [i[2] for i in x_list]

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x_listx, x_listy, x_listz, 'bo', label='X vector rotation')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')

    plt.show()

def plot_3DZ(att, ti, tf, dt):
    """
    %run: plot_3DZ(att, 0, 365*5, 0.1)

    :param att: attitude object
    :param ti: initial time [days]
    :param tf: final time [days]
    :param dt: step time for calculating the data point [days]
    :return: plot of the position of the z-axis (unitary) of the scanner wrt LMN frame.
    """
    if isinstance(att, ggs.Attitude) is False:
        raise TypeError('att is not an Attitude object.')
    if type(ti) not in [int, float]:
        raise TypeError('ti must be non-negative real numbers.')
    if type(tf) not in [int, float]:
        raise TypeError('tf must be non-negative real numbers.')
    if type(dt) not in [int, float]:
        raise TypeError('dt must be non-negative real numbers.')
    if ti < 0:
        raise ValueError('ti cannot be negative.')
    if tf <0:
        raise ValueError('tf cannot be negative.')
    if dt <0:
        raise ValueError('dt cannot be negative.')
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

def plot_longitud_latitude(att, ti, tf, dt):
    """
    L. Lindegren, SAG_LL_014, Figure 6.
    %run: plot_longitud_latitude(att, 0, 365*5, 3/24)

    :param att: attitude object
    :param ti: initial time [days]
    :param tf: final time [days]
    :param dt: step time for calculating the data point [days]
    :return: plots the longitud and latitude angles in degrees of the z-axis of the scanner
    with respect to the LMN frame.
    """
    if isinstance(att, ggs.Attitude) is False:
        raise TypeError('att is not an Attitude object.')
    if type(ti) not in [int, float]:
        raise TypeError('ti must be non-negative real numbers.')
    if type(tf) not in [int, float]:
        raise TypeError('tf must be non-negative real numbers.')
    if type(dt) not in [int, float]:
        raise TypeError('dt must be non-negative real numbers.')
    if ti < 0:
        raise ValueError('ti cannot be negative.')
    if tf <0:
        raise ValueError('tf cannot be negative.')
    if dt <0:
        raise ValueError('dt cannot be negative.')

    att.reset()
    att.create_storage(ti, tf, dt)
    long_list = [i[5]%(2*np.pi) for i in att.storage]
    alt_list = [i[6] for i in att.storage]

    plt.figure()
    plt.plot(np.degrees(long_list), np.degrees(alt_list), 'b.')
    plt.xlabel('Longitud [deg]')
    plt.ylabel('Lattitude [deg] ')

    plt.rcParams.update({'font.size': 22})
    plt.title('Revolving scanning')
    plt.show()

def plot_3DW(att, ti, tf, dt):
    """
    %run: plot_3DW(att, 0, 365*5, 0.1)

    :param att: attitude object
    :param ti: initial time [days]
    :param tf: final time [days]
    :param dt: step time for calculating the data point [days]
    :return: plot of the total inertia vector (unitary) of the scanner wrt LMN frame.
    """
    if isinstance(att, ggs.Attitude) is False:
        raise TypeError('att is not an Attitude object.')
    if type(ti) not in [int, float]:
        raise TypeError('ti must be non-negative real numbers.')
    if type(tf) not in [int, float]:
        raise TypeError('tf must be non-negative real numbers.')
    if type(dt) not in [int, float]:
        raise TypeError('dt must be non-negative real numbers.')
    if ti < 0:
        raise ValueError('ti cannot be negative.')
    if tf <0:
        raise ValueError('tf cannot be negative.')
    if dt <0:
        raise ValueError('dt cannot be negative.')
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
    """
    %run: plot_3Dsun(att, 0, 365*5, 0.1)

    :param att: attitude object
    :param ti: initial time [days]
    :param tf: final time [days]
    :param dt: step time for calculating the data point [days]
    :return: plot of the position of the sun wrt lmn frame. (circular orbit of 1AU)
    """
    if isinstance(att, ggs.Attitude) is False:
        raise TypeError('att is not an Attitude object.')
    if type(ti) not in [int, float]:
        raise TypeError('ti must be non-negative real numbers.')
    if type(tf) not in [int, float]:
        raise TypeError('tf must be non-negative real numbers.')
    if type(dt) not in [int, float]:
        raise TypeError('dt must be non-negative real numbers.')
    if ti < 0:
        raise ValueError('ti cannot be negative.')
    if tf <0:
        raise ValueError('tf cannot be negative.')
    if dt <0:
        raise ValueError('dt cannot be negative.')
    att.reset()
    att.create_storage(ti, tf, dt)

    sun_list = [obj[7] for obj in att.storage]

    x_srs_ecliptical = [i[0] for i in sun_list]
    y_srs_ecliptical = [i[1] for i in sun_list]
    z_srs_ecliptical = [i[2] for i in sun_list]

    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x_srs_ecliptical, y_srs_ecliptical, z_srs_ecliptical, 'bo--', label='s vector /sun movement')

    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.show()

def plot_attitude(att, ti, tf, dt):
    """
    L.Lindegren, SAG_LL_35, Figure 1.
    %run: plot_attitude(att, 0, 80, 0.01)
    L.Lindegren, SAG_LL_35, Figure 2.
    %run: plot_attitude(att, 0, 1, 0.01)

    :param att: attitude object
    :param ti: initial time [days]
    :param tf: final time [days]
    :param dt: step time [days]
    :return: plot of the 4 components of the attitude quaternion of the scanner,
    following the Nominal Scanning Law (NSL).
    """
    if isinstance(att, ggs.Attitude) is False:
        raise TypeError('att is not an Attitude object.')
    if type(ti) not in [int, float]:
        raise TypeError('ti must be non-negative real numbers.')
    if type(tf) not in [int, float]:
        raise TypeError('tf must be non-negative real numbers.')
    if type(dt) not in [int, float]:
        raise TypeError('dt must be non-negative real numbers.')
    if ti < 0:
        raise ValueError('ti cannot be negative.')
    if tf <0:
        raise ValueError('tf cannot be negative.')
    if dt <0:
        raise ValueError('dt cannot be negative.')
    att.reset()
    att.create_storage(ti, tf, dt)

    times= np.arange(ti, tf, dt)
    attitudes = [obj[4] for obj in att.storage]

    qw_list = [obj.w for obj in attitudes]
    qx_list = [obj.x for obj in attitudes]
    qy_list = [obj.y for obj in attitudes]
    qz_list = [obj.z for obj in attitudes]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.subplots_adjust(left=0.2, wspace=0.6)

    ax1.plot(times, qw_list, 'r--')
    ax1.set(title='W', xlabel='days')

    ax2.plot(times, qx_list, 'b--')
    ax2.set(title='X', xlabel='days')

    ax3.plot(times, qy_list, 'g--')
    ax3.set(title='Y', xlabel='days')

    ax4.plot(times, qz_list, 'k--')
    ax4.set(title='Z', xlabel='days')

    plt.rcParams.update({'font.size': 22})

    plt.show()

def plot_observations(scanner, sky):
    """
    :param scanner: scanner object
    :param sky: sky object
    :return: plots the observations made by the scanner.
    """
    if isinstance(scanner, ggs.Scanner) is False:
        raise TypeError('scanner is not an Scanner object.')
    if isinstance(sky, ggs.Sky) is False:
        raise TypeError('sky is not an Sky object.')

    x = [i[0] for i in scanner.telescope_positions]
    y = [i[1] for i in scanner.telescope_positions]
    z = [i[2] for i in scanner.telescope_positions]
    
    x_star = [i.coor[0] for i in sky.elements]
    y_star = [i.coor[1] for i in sky.elements]
    z_star = [i.coor[2] for i in sky.elements]

    fig1, (ax1, ax2,  ax3) = plt.subplots(1, 3)
    fig1.subplots_adjust(left=0.2, wspace=0.6)
    
    ax1.plot(x, y,'ro')
    ax1.plot(x_star, y_star, 'b*', ms=10)
    ax1.set(title='XY PLANE')

    ax2.plot(y, z, 'ko')
    ax2.plot(y_star, z_star, 'b*', ms=10)
    ax2.set(title='YZ PLANE')

    ax3.plot(x, z, 'go')
    ax3.plot(x_star, z_star, 'b*', ms=10)
    ax3.set(title='XZ PLANE')


    mpl.rcParams['legend.fontsize'] = 10

    fig2 = plt.figure()
    ax = fig2.gca(projection='3d')

    ax.plot(x, y, z, 'b*', label='observation measurements')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.show()

def plot_sky(sky):
    """
    :param sky: Sky object.
    :return: plot of the positions in 3D of the stars in sky.
    """
    x_list = [star.coor[0] for star in sky.elements]
    y_list = [star.coor[1] for star in sky.elements]
    z_list = [star.coor[2] for star in sky.elements]

    mpl.rcParams['legend.fontsize'] = 10

    fig2 = plt.figure()
    ax = fig2.gca(projection='3d')

    ax.plot(x_list, y_list, z_list, 'b*', label='Position of stars')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')
    plt.show()