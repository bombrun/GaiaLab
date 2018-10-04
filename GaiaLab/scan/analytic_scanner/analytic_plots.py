# -*- coding: utf-8 -*-

"""
Created on Mon Jun 18 14:59:19 2018

@author: mdelvallevaro

modified by: LucaZampieri

Plot helper functions, and other helper functions
"""

# # Imports
# Global imports
import matplotlib as mpl
from matplotlib import collections as mc
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d

# Local imports
import constants as const
import helpers as helpers
import frame_transformations as ft
from quaternion import Quaternion
from source import Source
from satellite import Satellite
from scanner import Scanner


def plot_attitude(sat, ti, tf, n_points=1000, figsize=(9, 5)):
    """
    L.Lindegren, SAG_LL_35, Figure 1.
    %run: plot_Satellite(sat, 0, 80, 0.01)
    L.Lindegren, SAG_LL_35, Figure 2.
    %run: plot_Satellite(sat, 0, 1, 0.01)

    :param sat: gaia satellite, Satellite object
    :param ti: initial time [days]
    :param tf: final time [days]
    :param n_points: number of points to be plotted of the function
    :return:
    Plot of the 4 components of the attitude of the satellite.
    attitude = (t, x, y, z)
    Each graph plots time in days versus each component evolution wrt time.

    note: the difference between this function and the function under the same
     name in the file geometric_plots lies on the calculation of the attitude.
     Here the points plotted are calculated from the spline, in contrast with
     the numerical methods calculation for geometric_plots.plot_attitude
     function.
    """
    if isinstance(sat, Satellite) is False:
        raise TypeError('sat is not an Satellite object.')
    times = np.linspace(ti, tf, n_points)
    attitudes = [sat.func_attitude(t) for t in times]

    qw_list = [obj.w for obj in attitudes]
    qx_list = [obj.x for obj in attitudes]
    qy_list = [obj.y for obj in attitudes]
    qz_list = [obj.z for obj in attitudes]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.subplots_adjust(left=0.2, wspace=0.6, hspace=1.5)

    ax1.plot(times, qw_list, 'r-')
    ax1.set(title='W', xlabel='days', xlim=(ti, tf),
            xticks=(np.arange(ti, tf+1, (tf+1-ti)//4)))

    ax2.plot(times, qx_list, 'b-')
    ax2.set(title='X', xlabel='days', xlim=(ti, tf),
            xticks=(np.arange(ti, tf+1, (tf+1-ti)//4)))

    ax3.plot(times, qy_list, 'g-')
    ax3.set(title='Y', xlabel='days', xlim=(ti, tf),
            xticks=(np.arange(ti, tf+1, (tf+1-ti)//4)))

    ax4.plot(times, qz_list, 'k-')
    ax4.set(title='Z', xlabel='days', xlim=(ti, tf),
            xticks=(np.arange(ti, tf+1, (tf+1-ti)//4)))

    plt.rcParams.update({'font.size': 22})
    plt.show()


def plot_observations(source, satellite, scan):
    """
    :param source: source scanned (object)
    :param satellite: Satellite object
    :param scan: scan object
    :return: plot of position of observations and their error bars.
    """

    if isinstance(satellite, Satellite) is False:
        raise TypeError('satellite is not an Satellite object.')
    if isinstance(scan, Scanner) is False:
        raise TypeError('scan is not an Scanner object.')

    alphas_obs = []
    deltas_obs = []
    radius_obs = []

    star_alphas = []
    star_deltas = []
    star_radius = []

    z_alphas = []
    z_deltas = []

    plt.figure()

    # for each of the observed times we plot the position of the x-axis in lmn
    # of the scanner
    for i, t in enumerate(scan.obs_times):

        alpha, delta, radius = ft.vector_to_polar(satellite.func_x_axis_lmn(t))
        alphas_obs.append(alpha % (2 * np.pi))
        deltas_obs.append(delta)
        # radius_obs.append(radius)
        source.set_time(t)
        star_alphas.append(source.alpha)
        star_deltas.append(source.delta)
        # star_deltas.append(source.radius)

        xaxis = satellite.func_x_axis_lmn(t)
        zaxis = satellite.func_z_axis_lmn(t)

        vectorz1 = xaxis + scan.z_threshold * zaxis
        vectorz2 = xaxis - scan.z_threshold * zaxis

        z_alpha_1, z_delta_1, z_radius_1 = ft.vector_to_polar(vectorz1)
        z_alpha_2, z_delta_2, z_radius_2 = ft.vector_to_polar(vectorz2)

        z_alphas.append([z_alpha_1, z_alpha_2])
        z_deltas.append([z_delta_1, z_delta_2])

    # For each couple of ([alpha1,alpha2],[delta1,delta2])
    for alpha_delta in zip(z_alphas, z_deltas):
        plt.plot(alpha_delta[0], alpha_delta[1], 'yo-')

    plt.plot(alphas_obs, deltas_obs, 'ro', label='observations')  # plot observation as re dots
    plt.plot(star_alphas, star_deltas, 'b*', label='star')  # plot stars as blu stars

    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.legend(loc='upper left')
    plt.title('%s' % source.name)
    plt.xlabel('alpha [rad]')
    plt.ylabel('delta [rad]')
    plt.axis('equal')
    plt.tight_layout()
    plt.margins(0.1)
    plt.show()


def plot_prediction_VS_reality(source, satellite, scan, num_observations=0, angle_tolerance=0.1):
    """
    :param source: source scanned (object)
    :param satellite: Satellite object
    :param scan: scan object
    :param num_observations: number of observation we want to plot
    :param
    :return: plot of position of observations and their error bars.
    """

    if isinstance(satellite, Satellite) is False:
        raise TypeError('satellite is not an Satellite object.')
    if isinstance(scan, Scanner) is False:
        raise TypeError('scan is not an Scanner object.')

    alphas_obs = []
    deltas_obs = []
    radius_obs = []
    star_alphas = []
    star_deltas = []
    star_radius = []
    z_alphas = []
    z_deltas = []
    predictions_alphas = []
    predictions_deltas = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Set the number of observations we want to plot
    if num_observations != 0:
        obs_times = scan.obs_times[0:num_observations]
    else:
        obs_times = scan.obs_times
    # for each of the observed times we plot the position of the x-axis in lmn
    # of the scanner
    for i, t in enumerate(obs_times):

        alpha, delta, radius = ft.vector_to_polar(satellite.func_x_axis_lmn(t))
        alphas_obs.append(alpha % (2 * np.pi))
        deltas_obs.append(delta)
        # radius_obs.append(radius)
        source.set_time(t)
        star_alphas.append(source.alpha)
        star_deltas.append(source.delta)
        # star_deltas.append(source.radius)

        # Axis of the satellite in the lmn frame
        xaxis = satellite.func_x_axis_lmn(t)
        zaxis = satellite.func_z_axis_lmn(t)

        # Vectors describing the endpoints of the interval in which the source must be
        # first in the lmn frame then in the polar one
        vectorz1 = xaxis + scan.z_threshold * zaxis
        z_alpha_1, z_delta_1, z_radius_1 = ft.vector_to_polar(vectorz1)
        vectorz2 = xaxis - scan.z_threshold * zaxis
        z_alpha_2, z_delta_2, z_radius_2 = ft.vector_to_polar(vectorz2)

        z_alphas.append([z_alpha_1, z_alpha_2])
        z_deltas.append([z_delta_1, z_delta_2])

    # For each couple of ([alpha1,alpha2],[delta1,delta2])
    alphas_deltas = list(zip(z_alphas, z_deltas))
    for i in range(len(alphas_deltas)-1):
        x1_x2, y1_y2 = alphas_deltas[i]
        x3_x4, y3_y4 = alphas_deltas[i+1]

        # compute dot product between the two segments
        dot_product = np.dot([x1_x2[0]-x1_x2[1], y1_y2[0]-y1_y2[1]],
                             [x3_x4[0]-x3_x4[1], y3_y4[0]-y3_y4[1]])
        angle = helpers.compute_angle([x1_x2[0]-x1_x2[1], y1_y2[0]-y1_y2[1]],
                                      [x3_x4[0]-x3_x4[1], y3_y4[0]-y3_y4[1]])
        # print('angle: ', angle)
        if angle < angle_tolerance:
            continue
        # compute the intersection between observations
        intersection, error_msg = helpers.compute_intersection(x1_x2[0], y1_y2[0],
                                                               x1_x2[1], y1_y2[1],
                                                               x3_x4[0], y3_y4[0],
                                                               x3_x4[1], y3_y4[1])
        # print(error_msg)

        if not error_msg:
            predictions_alphas.append(intersection[0])
            predictions_deltas.append(intersection[1])

    # Plot the prediction of the positions of the stars as being the intersection of
    # the error bands of the observations
    # print(predictions_alphas, '\n  -----  \n', predictions_deltas)
    for ax in (ax1, ax2):
        ax.plot(predictions_alphas, predictions_deltas, 'rx:', label='predictions')
        # plot stars as blu stars
        ax.plot(star_alphas, star_deltas, 'b*:', label='star')
        # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.set_title('%s' % source.name)
        ax.set_xlabel('alpha [rad]')
        ax.set_ylabel('delta [rad]')
        ax.axis('equal')
        ax.grid()
        # ax.set_tight_layout()
        ax.margins(0.1)

    for i, alpha_delta in enumerate(zip(z_alphas, z_deltas)):
        ax1.plot(alpha_delta[0], alpha_delta[1], 'o-', label='obs_'+str(i))  # ,'yo-'

    ax1.legend()
    ax2.legend()
    return fig


def plot_phi(source, sat, ti=0, tf=90, n=1000):
    styles = ['b.--', 'r.']
    times_total = np.linspace(ti, tf, n)
    phi_list = []
    eta_list = []
    for t in times_total:
        phi_value, eta_value = phi(source, sat, t)
        eta_list.append(eta_value)
        phi_list.append(phi_value)

    fig1 = plt.figure(1)
    plt.plot(times_total, phi_list, styles[0])
    plt.hlines(0, xmin=times_total[0], xmax=times_total[-1], color='g')
    plt.xlabel('time [days]')
    plt.ylabel('Phi [rad]')

    fig2 = plt.figure(2)
    plt.plot(times_total, eta_list, styles[1])
    plt.hlines(0, xmin=times_total[0], xmax=times_total[-1], color='g')
    plt.xlabel('time [days]')
    plt.ylabel('Eta[rad]')

    return fig1, fig2


def plot_eta_over_phi(source, sat, ti=0, tf=90, n=1000):
    times_total = np.linspace(ti, tf, n)
    phi_list = []
    eta_list = []
    for t in times_total:
        phi_value, eta_value = phi(source, sat, t)
        eta_list.append(eta_value)
        phi_list.append(phi_value)

    plt.figure(1)
    plt.plot(phi_list, eta_list, 'b,')
    plt.xlabel('Phi [rad]')
    plt.ylabel('Eta [rad]')

    plt.show()


def plot_eta_over_phi_day(source, sat, ti=0, tf=90, n=1000, day=45):
    times_total = np.linspace(ti, tf, n)
    phi_list = []
    eta_list = []
    for t in times_total:
        phi_value, eta_value = phi(source, sat, t)
        eta_list.append(eta_value)
        phi_list.append(phi_value)
    phi_actual, eta_actual = phi(source, sat, day)

    p = plt.figure(1)
    plt.plot(phi_list, eta_list, 'b,')
    plt.plot(phi_actual, eta_actual, 'bo')
    plt.xlabel('Phi [rad]')
    plt.ylabel('Eta [rad]')

    return p


def plot_stars_trajectory(source, satellite):
    """
    :param source: source object
    :param satellite: Satellite object
    :param t_total: total time for which the trajectory is desired [days] from
     J2000.
    :return: plot of the star trajectory in the lmn-frame.
    """
    if isinstance(source, Source) is False:
        raise TypeError('source is not an Source object.')
    if isinstance(satellite, Satellite) is False:
        raise TypeError('satellite is not an Satellite object.')

    time_total = satellite.storage[-1][0]

    alphas = []
    deltas = []
    for i in np.arange(0, time_total, 1):
        alpha_obs, delta_obs, delta_alpha_dx_mas, delta_delta_mas = source.topocentric_angles(satellite, i)
        alphas.append(delta_alpha_dx_mas)
        deltas.append(delta_delta_mas)

    n = len(alphas)
    times = np.linspace(2000, 2000 + time_total/const.days_per_year, n)

    # Styles for the plots
    path_style = 'b:s'
    origin_style = 'kx'

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(121)

    fig.suptitle(r'$\varpi={%.2f}$, $\mu_{{\alpha*}}={%.2f}$, $\mu_\delta={%.2f}$'
                 % (source.parallax, source.mu_alpha_dx, source.mu_delta),
                 fontsize=20)

    ax.plot(alphas, deltas, path_style,
            label=r'%s path' % (source.name), lw=2)
    ax.plot(alphas[0], deltas[0], origin_style, label='origin')
    ax.set_xlabel(r'$\Delta\alpha*$ [mas]')
    ax.set_ylabel(r'$\Delta\delta$ [mas]')
    ax.axhline(y=0, c='gray', lw=1)
    ax.axvline(x=0, c='gray', lw=1)
    ax.legend(fontsize=12, facecolor='#000000',
              framealpha=0.1)
    # Top right subplot
    ax1dra = fig.add_subplot(222)
    ax1dra.plot(times, alphas, path_style)
    ax1dra.plot(times[0], alphas[0], origin_style, label='origin')
    ax1dra.axhline(y=0, c='gray', lw=1)
    # ax1dra.set_xlabel(r'Time [yr]')
    ax1dra.set_ylabel(r'$\Delta\alpha*$ [mas]')

    # Top left subplot
    ax1ddec = fig.add_subplot(224, sharex=ax1dra)
    ax1ddec.axhline(y=0, c='gray', lw=1)
    ax1ddec.plot(times, deltas, path_style)
    ax1ddec.plot(times[0], deltas[0], origin_style, label='origin')
    ax1ddec.set_xlabel(r'Time [yr]')
    ax1ddec.set_ylabel(r'$\Delta\delta$ [mas]')

    # plt.tight_layout()
    plt.show()


def plot_stars_trajectory_3D(source, satellite):
    """
    :param source: source object
    :param satellite: Satellite object
    :param t_total: total time for which the trajectory is desired [days] from
     J2000.
    :return: plot of the star trajectory in the lmn-frame.
    """
    if isinstance(source, Source) is False:
        raise TypeError('source is not an Source object.')
    if isinstance(satellite, Satellite) is False:
        raise TypeError('satellite is not an Satellite object.')

    time_total = satellite.storage[-1][0]

    alphas = []
    deltas = []
    for i in np.arange(0, time_total, 1):
        alpha_obs, delta_obs, delta_alpha_dx_mas, delta_delta_mas = source.topocentric_angles(satellite, i)
        alphas.append(delta_alpha_dx_mas)
        deltas.append(delta_delta_mas)

    n = len(alphas)
    times = np.linspace(2000, 2000 + time_total/const.days_per_year, n)

    # Styles for the plots
    path_style = 'b:s'
    origin_style = 'kx'

    fig = plt.figure(figsize=(16, 9))
    ax = fig.gca(projection='3d')

    fig.suptitle(r'$\varpi={%.2f}$, $\mu_{{\alpha*}}={%.2f}$, $\mu_\delta={%.2f}$'
                 % (source.parallax, source.mu_alpha_dx, source.mu_delta),
                 fontsize=20)

    ax.plot(alphas, deltas, times, path_style,
            label=r'%s path' % (source.name), lw=2)
    # ax.plot(alphas[0], deltas[0], times[0], origin_style, label='origin')
    ax.set_xlabel(r'$\Delta\alpha*$ [mas]')
    ax.set_ylabel(r'$\Delta\delta$ [mas]')
    ax.axhline(y=0, c='gray', lw=1)
    ax.axvline(x=0, c='gray', lw=1)
    ax.legend(fontsize=12, facecolor='#000000',
              framealpha=0.1)
    # plt.tight_layout()
    plt.show()


def plot_3D_scanner_pos(sat, axis, ti, tf, n_points=1000, elevation=10, azimuth=10):
    """
    %run: plot_3D_scanner_pos(sat, 'X', 0, 365*5, 0.1)

    :param sat: Satellite object
    :param ti: initial time [float][days]
    :param tf: final time [float][days]
    :param n_points: number of points to be plotted [int]
    :param elevation: (plot aestetic) set the elevation of the 3D view
    :param azimuth: (plot aestetic) set the azimuth of the 3D view
    :return: plot of the position of the given axis (unitary) of the scanner wrt
     LMN frame.
    """
    if isinstance(sat, Satellite) is False:
        raise TypeError('sat is not an Satellite object.')
    if type(ti) not in [int, float]:
        raise TypeError('ti must be non-negative real numbers.')
    if type(tf) not in [int, float]:
        raise TypeError('tf must be non-negative real numbers.')
    if type(n_points) not in [int, float]:
        raise TypeError('dt must be non-negative real numbers.')
    if axis not in ['X', 'Z']:
        raise ValueError("Axis can be either 'X' or 'Z'.")
    if ti < 0:
        raise ValueError('ti cannot be negative.')
    if tf < 0:
        raise ValueError('tf cannot be negative.')
    if n_points < 0:
        raise ValueError('dt cannot be negative.')

    times = np.linspace(ti, tf, n_points)
    if axis == 'X':
        axis_list = [sat.func_x_axis_lmn(t) for t in times]
        label_ = 'X vector rotation'
        style_ = 'b,'
    elif axis == 'Z':
        axis_list = [sat.func_z_axis_lmn(t) for t in times]
        label_ = 'Z vector rotation'
        style_ = 'r,'

    listx = [i[0] for i in axis_list]
    listy = [i[1] for i in axis_list]
    listz = [i[2] for i in axis_list]

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(listx, listy, listz, style_, label=label_)
    ax.plot([0], [0], [0], 'kx', label='origin')
    ax.legend()
    ax.set_xlabel('L')
    ax.set_ylabel('M')
    ax.set_zlabel('N')
    ax.azim = azimuth
    ax.elev = elevation

    plt.show()


# Draft of helper functions
def phi(source, sat, t):
    """
    Calculates the diference between the x-axis of the satellite and the direction vector to the star.
    Once this is calculated, it checks how far away is in the alpha direction (i.e. the y-component) wrt IRS.
    :param source: Source [object]
    :param sat: Satellite [object]
    :param t: time [float][days]
    :return: [float] angle, alpha wrt IRS.
    """
    t = float(t)
    u_lmn_unit = source.unit_topocentric_function(sat, t)
    phi_value_lmn = u_lmn_unit - sat.func_x_axis_lmn(t)
    phi_value_xyz = ft.lmn_to_xyz(sat.func_attitude(t), phi_value_lmn)
    phi = np.arcsin(phi_value_xyz[1])
    eta = np.arcsin(phi_value_xyz[2])
    return phi, eta

################################################################################
# Undefined functions


def run():
    """
    Create the objects source for Sirio, Vega and Proxima as well
    as the corresponding scanners and the satellite object of Gaia.
    Then scan the sources from Gaia and print the time.
    :return: gaia, sirio, scanSirio, vega, scanVega, proxima, scanProxima
    """
    start_time = time.time()
    sirio = Source("sirio", 101.28, -16.7161, 379.21, -546.05, -1223.14, -7.6)
    vega = Source("vega", 279.2333, 38.78, 128.91, 201.03, 286.23, -13.9)
    proxima = Source("proxima", 217.42, -62, 768.7, 3775.40, 769.33, 21.7)

    scanSirio = Scanner(np.radians(20), np.radians(2))
    scanVega = Scanner(np.radians(20), np.radians(2))
    scanProxima = Scanner(np.radians(20), np.radians(2))
    gaia = Satellite()
    print(time.time() - start_time)

    scanSirio.start(gaia, sirio)
    scanVega.start(gaia, vega)
    scanProxima.start(gaia, proxima)
    print(time.time() - start_time)

    seconds = time.time() - start_time
    print('Total seconds:', seconds)
    return gaia, sirio, scanSirio, vega, scanVega, proxima, scanProxima


################################################################################
# # isInstance functions
# this function should not be used
def test_is_satellite(other):
    """ Tests if (other) is of type satellite. Raise exception otherwise.
    """
    if not isinstance(other, Satellite):
        raise TypeError('{} is not an Satellite object'.format(type(other)))
    else:
        pass


# not used yet
def test_object_type(other, type_str):
    """
    Tests if (other) is of type (type_str). Raise exception otherwise.
    :param other: Variable which type should be tested
    :param type_str: [str] string containing the object type we want to test.
    """

    possible_types = {"Source": Source,
                      "Satellite": Satellite,
                      "Satellite": Satellite,
                      "Scanner": Scanner}
    if type_str not in possible_types:
        raise TypeError('Expected type "{}" is not part of the possible_types'.format(type_str))

    expected_type = possible_types[type_str]

    if not isinstance(other, expected_type):
        raise TypeError('Type "{}" is not "{}"'.format(type(other), type_str))
