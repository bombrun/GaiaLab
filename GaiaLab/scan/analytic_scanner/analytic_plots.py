# -*- coding: utf-8 -*-


import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from quaternion import Quaternion
import frame_transformations as ft
import gaia_analytic_toymodel as ggs

def plot_3DX(att, ti, tf, n_points = 1000):
    """
    %run: plot_3DX(att, 0, 365*5, 0.1)

    :param att: attitude object
    :param ti: initial time [float][days]
    :param tf: final time [float][days]
    :param n_points: number of points to be plotted [int]
    :return: plot of the position of the x-axis (unitary) of the scanner wrt LMN frame.
    """
    if isinstance(att, ggs.Attitude) is False:
        raise TypeError('att is not an Attitude object.')
    if type(ti) not in [int, float]:
        raise TypeError('ti must be non-negative real numbers.')
    if type(tf) not in [int, float]:
        raise TypeError('tf must be non-negative real numbers.')
    if type(n_points) not in [int, float]:
        raise TypeError('dt must be non-negative real numbers.')
    if ti < 0:
        raise ValueError('ti cannot be negative.')
    if tf <0:
        raise ValueError('tf cannot be negative.')
    if n_points <0:
        raise ValueError('dt cannot be negative.')

    times = np.linspace(ti, tf, n_points)
    x_list = [att.func_x_axis_lmn(t) for t in times]

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

def plot_3DZ(att, ti, tf, n_points = 1000):
    """
    %run: plot_3DZ(att, 0, 365*5, 0.1)

    :param att: attitude object
    :param ti: initial time [days]
    :param tf: final time [days]
    :param n_points: number of points to be plotted [int]
    :return: plot of the position of the z-axis (unitary) of the scanner wrt LMN frame.
    """
    if isinstance(att, ggs.Attitude) is False:
        raise TypeError('att is not an Attitude object.')
    if type(ti) not in [int, float]:
        raise TypeError('ti must be non-negative real numbers.')
    if type(tf) not in [int, float]:
        raise TypeError('tf must be non-negative real numbers.')
    if type(n_points) not in [int, float]:
        raise TypeError('dt must be non-negative real numbers.')
    if ti < 0:
        raise ValueError('ti cannot be negative.')
    if tf <0:
        raise ValueError('tf cannot be negative.')
    if n_points <0:
        raise ValueError('dt cannot be negative.')

    times = np.linspace(ti, tf, n_points)
    z_list = [att.func_z_axis_lmn(t) for t in times]

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


def plot_attitude(att, ti, tf, n_points=1000):
    """
    L.Lindegren, SAG_LL_35, Figure 1.
    %run: plot_attitude(att, 0, 80, 0.01)
    L.Lindegren, SAG_LL_35, Figure 2.
    %run: plot_attitude(att, 0, 1, 0.01)

    :param att: gaia satellite, attitude object
    :param ti: initial time [days]
    :param tf: final time [days]
    :param n_points: number of points to be plotted of the function
    :return:
    Plot of the 4 components of the attitude of the satellite.
    attitude = (t, x, y, z)
    Each graph plots time in days versus each component evolution wrt time.

    note: the difference between this function and the function under the same name in the file geometric_plots
    lies on the calculation of the attitude. Here the points plotted are calculated from the spline, in contrast
    with the numerical methods calculation for geometric_plots.plot_attitude function.
    """
    if isinstance(att, ggs.Attitude) is False:
        raise TypeError('att is not an Attitude object.')
    times= np.linspace(ti, tf, n_points)
    attitudes = [att.func_attitude(t) for t in times]

    qw_list = [obj.w for obj in attitudes]
    qx_list = [obj.x for obj in attitudes]
    qy_list = [obj.y for obj in attitudes]
    qz_list = [obj.z for obj in attitudes]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.subplots_adjust(left=0.2, wspace=0.6)

    ax1.plot(times, qw_list, 'r-')
    ax1.set(title='W', xlabel='days')

    ax2.plot(times, qx_list, 'b-')
    ax2.set(title='X', xlabel='days')

    ax3.plot(times, qy_list, 'g-')
    ax3.set(title='Y', xlabel='days')

    ax4.plot(times, qz_list, 'k-')
    ax4.set(title='Z', xlabel='days')

    plt.rcParams.update({'font.size': 22})

    plt.show()

from matplotlib import collections as mc

def plot_observations(source, satellite, scan):
    """
    :param source: source scanned (object)
    :param satellite: Attitude object
    :param scan: scan object
    :return: plot of position of observations and their error bars.
    """

    if isinstance(satellite, ggs.Attitude) is False:
        raise TypeError('satellite is not an Attitude object.')
    if isinstance(scan, ggs.Scanner) is False:
        raise TypeError('scan is not an Scanner object.')

    alphas_obs = []
    deltas_obs = []
    star_alphas = []
    star_deltas = []

    plt.figure()

    #for each of the observed times we plot the position of the x-axis in lmn of the scanner,
    # and the error is equivalent to the z-threshold values and the y-threshold values.
    for t in scan.obs_times:
        zalphas = []
        zdeltas = []

        alpha, delta, radius = ft.to_polar(satellite.func_x_axis_lmn(t))
        alphas_obs.append(alpha% (2 * np.pi))
        deltas_obs.append(delta)
        source.set_time(t)
        star_alphas.append(source.alpha)
        star_deltas.append(source.delta)

        xaxis = satellite.func_x_axis_lmn(t)
        zaxis = satellite.func_z_axis_lmn(t)

        vectorz1 = xaxis + scan.z_threshold * zaxis
        vectorz2 = xaxis - scan.z_threshold * zaxis

        z_alpha_1, z_delta_1, z_radius_1 = ft.to_polar(vectorz1)
        z_alpha_2, z_delta_2, z_radius_2 = ft.to_polar(vectorz2)
        zalphas.append(z_alpha_1)
        zalphas.append(z_alpha_2)
        zdeltas.append(z_delta_1)
        zdeltas.append(z_delta_2)
        plt.plot(zalphas, zdeltas, 'yo-')

    plt.plot(alphas_obs, deltas_obs, 'ro', label = 'observations')
    plt.plot(star_alphas, star_deltas, 'b*', label = 'star')
    plt.legend(loc = 'upper left')
    plt.title('%s' %source.name)
    plt.xlabel('alpha [rad]')
    plt.ylabel('delta [rad]')
    plt.axis('equal')
    plt.tight_layout()
    plt.margins(0.1)
    plt.show()

def plot_phi(source, att, ti = 0, tf =90, n = 1000):
    times_total= np.linspace(ti, tf, n)
    phi_list = []
    eta_list = []
    for t in times_total:
        phi_value, eta_value = ggs.phi(source, att, t)
        eta_list.append(eta_value)
        phi_list.append(phi_value)

    plt.figure(1)
    plt.plot(times_total, phi_list, 'bo-')
    plt.hlines(0, xmin = times_total[0], xmax = times_total[-1], color = 'g')
    plt.xlabel('time [days]')
    plt.ylabel('Phi [rad]')

    plt.figure(2)
    plt.plot(times_total, eta_list, 'ro-')
    plt.hlines(0, xmin = times_total[0], xmax = times_total[-1], color = 'g')
    plt.xlabel('time [days]')
    plt.ylabel('Eta[rad]')
    plt.show()

def plot_stars_trajectory(source, satellite):
    """
    :param source: source object
    :param satellite: attitude object
    :param t_total: total time for which the trajectory is desired [days] from J2000.
    :return: plot of the star trajectory in the lmn-frame.
    """
    if isinstance(source, ggs.Source) is False:
        raise TypeError('source is not an Source object.')
    if isinstance(satellite, ggs.Attitude) is False:
        raise TypeError('satellite is not an Attitude object.')

    time_total = satellite.storage[-1][0]

    alphas = []
    deltas = []
    for i in np.arange(0, time_total, 1):
        alpha_obs, delta_obs, delta_alpha_dx_mas, delta_delta_mas = source.topocentric_angles(satellite, i)
        alphas.append(delta_alpha_dx_mas)
        deltas.append(delta_delta_mas)

    n = len(alphas)
    times = np.linspace(2000, 2000 + time_total/365, n)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(121)

    ax.plot(alphas, deltas,'b-',
            label=r'%s path' %(source.name), lw=2)
    ax.set_xlabel(r'$\Delta\alpha*$ [mas]')
    ax.set_ylabel(r'$\Delta\delta$ [mas]')
    ax.axhline(y=0, c='gray', lw=1)
    ax.axvline(x=0, c='gray', lw=1)
    ax.legend(loc='upper right', fontsize=12, facecolor='#000000', framealpha=0.1)
    ax.set_title(r'$\varpi={%.2f}$, $\mu_{{\alpha*}}={%.2f}$, $\mu_\delta={%.2f}$'
              %(source.parallax, source.mu_alpha_dx, source.mu_delta))

    ax1dra = fig.add_subplot(222)
    ax1dra.axhline(y=0, c='gray', lw=1)
    ax1dra.set_xlabel(r'Time [yr]')
    ax1dra.plot(times, alphas, 'b-')
    ax1dra.set_ylabel(r'$\Delta\alpha*$ [mas]')

    ax1ddec = fig.add_subplot(224)
    ax1ddec.axhline(y=0, c='gray', lw=1)
    ax1ddec.plot(times, deltas, 'b-')
    ax1ddec.set_xlabel(r'Time [yr]')
    ax1ddec.set_ylabel(r'$\Delta\delta$ [mas]')

    plt.tight_layout()
    plt.show()

