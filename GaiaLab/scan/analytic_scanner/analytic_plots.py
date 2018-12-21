# -*- coding: utf-8 -*-
"""
File analytic_plots.py

:purpose: store ploting functions

Created on Mon Jun 18 14:59:19 2018

:authors: mdelvallevaro, LucaZampieri

.. warning::
    In case of doubts on the functionality of these functions, please check by
    plotting manually what is desired.

.. todo::
    If changed, update this functions

"""
# ### Uncomment next two line if the file is put in notebooks/ folder
# import sys
# sys.path.append('../GaiaLab/scan/analytic_scanner')

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
import quaternion
from source import Source
from satellite import Satellite
from scanner import Scanner
from agis_functions import *


# ### For simple visualization (source & satellite) ----------------------------
def plot_sources_in_sky(sources, projection='hammer'):
    """
    :param sources: [list of sources] Sources we want to plot
    :param projection: [string] kind of projection we want to apply to the plot
    :action: plots the sources in the sky at time ```t=0```
    """
    plt.figure()
    plt.subplot(111, projection=projection)
    for i, s in enumerate(sources):
        plt.plot(ft.zero_to_two_pi_to_minus_pi_pi(np.array([s.alpha])), s.delta, '+', label=s.name)
    plt.title("Hammer Projection of the Sky")
    plt.legend(loc=9, bbox_to_anchor=(1.1, 1))
    plt.grid(True)
    plt.show()


def plot_attitude(sat, ti, tf, n_points=1000, figsize=(9, 5), style='.--'):
    """
    Recreating the plot of L.Lindegren, SAG_LL_35:

    - Figure 1, run ``plot_Satellite(sat, 0, 80, 0.01)``
    - Figure 2, run ``plot_Satellite(sat, 0, 1, 0.01)``

    Each graph plots each component evolution wrt time.

    :param sat: gaia satellite, Satellite object
    :param ti: [float][days] initial time
    :param tf: [float][days] final time
    :param n_points: number of points to be plotted of the function
    :action: Plot of the 4 components of the attitude of the satellite.
             attitude = (t, x, y, z)
    """
    times = np.linspace(ti, tf, n_points)
    attitudes = [sat.func_attitude(t) for t in times]

    qw_list = [obj.w for obj in attitudes]
    qx_list = [obj.x for obj in attitudes]
    qy_list = [obj.y for obj in attitudes]
    qz_list = [obj.z for obj in attitudes]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.subplots_adjust(left=0.2, wspace=0.6, hspace=1.5)

    ax1.plot(times, qw_list, 'r'+style)
    ax1.set(title='W', xlabel='days', xlim=(ti, tf),
            xticks=(np.arange(ti, tf+1, (tf+1-ti)//4)))

    ax2.plot(times, qx_list, 'b'+style)
    ax2.set(title='X', xlabel='days', xlim=(ti, tf),
            xticks=(np.arange(ti, tf+1, (tf+1-ti)//4)))

    ax3.plot(times, qy_list, 'g'+style)
    ax3.set(title='Y', xlabel='days', xlim=(ti, tf),
            xticks=(np.arange(ti, tf+1, (tf+1-ti)//4)))

    ax4.plot(times, qz_list, 'k'+style)
    ax4.set(title='Z', xlabel='days', xlim=(ti, tf),
            xticks=(np.arange(ti, tf+1, (tf+1-ti)//4)))

    plt.rcParams.update({'font.size': 22})
    plt.show()


# ### END for simple visualization (source & satellite) ########################


# ### For visualizing scan results ---------------------------------------------
def plot_star(source, satellite, obs_times):
    """
    Plot wrt the ICRS point of view

    :param source: source scanned (object)
    :param satellite: Satellite object
    :param obs_times: [list][days] observed times
    :return: plot of position of observations and their error bars.
    """

    alphas_obs = []
    deltas_obs = []
    radius_obs = []

    star_alphas = []
    star_deltas = []
    star_radius = []

    z_alphas = []
    z_deltas = []

    fig = plt.figure()

    for i, t in enumerate(obs_times):
        source.set_time(t)
        star_alphas.append(source.alpha / const.rad_per_mas)
        star_deltas.append(source.delta / const.rad_per_mas)

    plt.plot(star_alphas, star_deltas, 'b*', label='star')
    plt.plot([star_alphas[0], star_alphas[-1]], [star_deltas[0], star_deltas[-1]], 'r-', alpha=0.2)

    plt.legend(loc='upper left')
    plt.title('%s' % source.name)
    plt.xlabel('alpha [mas]')
    plt.ylabel('delta [mas]')
    plt.axis('equal')
    plt.tight_layout()
    plt.margins(0.1)
    return fig


def plot_star_trajectory_with_scans(sat, source, obs_times, num_ms_for_snapshot=2):
    """
    Plots the star trajectory

    :param sat: [Satellite object]
    :param source: [source object]
    :param obs_times: [list of floats] list with the observations times from
     scanner
    :param num_ms_for_snapshot: [float][milliseconds] time intervall in which we
     desire the scanner positions
    :returns: figure object with the plot
    """
    observed_times = np.sort(obs_times)
    fig = plt.figure()
    plt.title(source.name + ' path with scan directions')

    star_alphas, star_deltas = ([], [])
    green_alphas, green_deltas = ([], [])

    for i, t in enumerate(observed_times):

        # Real star parameters
        alpha_star, delta_star, _, _ = source.topocentric_angles(sat, t)
        star_alphas.append(alpha_star/const.rad_per_mas)  # converts in [mas]
        star_deltas.append(delta_star/const.rad_per_mas)

        # Observed star parameters
        vector_to_star_in_comrs_frame = get_obs_in_CoMRS(source, sat, t)
        green_alpha, green_delta = ft.vector_to_alpha_delta(vector_to_star_in_comrs_frame)
        green_alphas.append(green_alpha/const.rad_per_mas)
        green_deltas.append(green_delta/const.rad_per_mas)

        # Scanner position parameters
        my_as, my_ds = ([], [])
        half_interval = num_ms_for_snapshot * 1/24/60/60/1000  # 2ms
        for ti in np.linspace(t-half_interval, t+half_interval, num=100):
            my_vector = get_obs_in_CoMRS(source, sat, ti)
            my_a, my_d = ft.vector_to_alpha_delta(my_vector)
            my_as.append(my_a/const.rad_per_mas)
            my_ds.append(my_d/const.rad_per_mas)
        p1, = plt.plot(my_as, my_ds, 'r-', alpha=0.5)
        p2, = plt.plot(my_as[0], my_ds[0], 'r+')  # start of scan direction
        p3, = plt.plot(my_as[-1], my_ds[-1], 'r>')  # end of star direction

    p_green, = plt.plot(green_alphas, green_deltas, 'gx:')
    p_star, = plt.plot(star_alphas, star_deltas, 'b.:', alpha=0.5)  # plot stars as blu stars

    plt.legend(handles=[p1, p2, p3, p_green, p_star],
               labels=['discretized scanner position', str(num_ms_for_snapshot)+'ms before scan',
                       str(num_ms_for_snapshot)+'ms after scan', 'observations of star', 'star'],
               loc='best',
               bbox_to_anchor=(1, 1))
    # plt.title('%s' % source.name)
    plt.xlabel('alpha [mas]'), plt.ylabel('delta [mas]')
    # plt.axis('equal'), plt.tight_layout()
    plt.margins(0.1), plt.grid()
    plt.show()
    return fig


def plot_field_angles(source, sat, obs_times=[], ti=0, tf=90, n=1000, limit=False, double_telescope=True):
    styles = ['b,', 'rs']
    zeta_limit = np.radians(0.5)
    eta_limit = np.radians(0.5)
    y_limit = (-zeta_limit*10, zeta_limit*10)
    times_total = np.linspace(ti, tf, n)
    eta_list = []
    zeta_list = []
    zeta_sol_list = []
    eta_sol_list = []
    for t in times_total:
        attitude = sat.func_attitude(t)
        eta_value, zeta_value = observed_field_angles(source, attitude, sat, t, double_telescope)
        eta_list.append(eta_value)
        zeta_list.append(zeta_value)
    for t in obs_times:
        attitude = sat.func_attitude(t)
        eta_value, zeta_value = observed_field_angles(source, attitude, sat, t, double_telescope)
        eta_sol_list.append(eta_value)
        zeta_sol_list.append(zeta_value)

    zeta_fig = plt.figure(1)
    plt.plot(times_total, zeta_list, styles[0], label='zeta path', alpha=0.5)
    plt.plot(obs_times, zeta_sol_list, styles[1], label='solutions')
    plt.hlines(0, xmin=times_total[0], xmax=times_total[-1], color='g')
    plt.hlines(zeta_limit, xmin=times_total[0], xmax=times_total[-1], color='g', linestyle='dotted',
               label='field of view limitation (5°)')
    plt.hlines(-zeta_limit, xmin=times_total[0], xmax=times_total[-1], color='g', linestyle='dotted')
    plt.xlim(ti, tf)
    if limit:
        plt.ylim(y_limit)
    plt.xlabel('time [days]')
    plt.ylabel('zeta [rad]')
    plt.grid()
    plt.legend()

    eta_fig = plt.figure(2)
    plt.plot(times_total, eta_list, styles[0], label='eta path')
    plt.plot(obs_times, eta_sol_list, styles[1], label='solutions')
    plt.hlines(0, xmin=times_total[0], xmax=times_total[-1], color='g')
    plt.hlines(eta_limit, xmin=times_total[0], xmax=times_total[-1], color='g', linestyle='dotted',
               label='field of view limitation (5°)')
    plt.hlines(-eta_limit, xmin=times_total[0], xmax=times_total[-1], color='g', linestyle='dotted')
    plt.xlim(ti, tf)
    if limit:
        plt.ylim(y_limit)
    plt.xlabel('time [days]')
    plt.ylabel('Eta[rad]')
    plt.grid()
    plt.legend()

    return zeta_fig, eta_fig


def plot_scanner_position_over_source(source, sat, t_init=0, t_end=365, num_points_of_discretization=10000):
    fig, ax = plt.subplots(1)
    ax.set_title('Discretized passages of gaia telescopes')
    list_ra_PFoV, list_ra_FFoV, list_dec_PFoV, list_dec_FFoV = ([], [], [], [])
    for t in np.linspace(t_init, t_end, num=num_points_of_discretization):
        ra_PFoV, dec_PFoV, ra_FFoV, dec_FFoV = np.array(get_angular_FFoV_PFoV(sat, t))
        ra_PFoV, ra_FFoV = ft.zero_to_two_pi_to_minus_pi_pi(np.array([ra_PFoV, ra_FFoV]))
        # dec_PFoV, dec_FFoV = ft.transform_twoPi_into_halfPi(np.array([dec_PFoV, dec_FFoV]))
        list_ra_PFoV.append(ra_PFoV)
        list_dec_PFoV.append(dec_PFoV)
        list_ra_FFoV.append(ra_PFoV)
        list_dec_FFoV.append(dec_FFoV)

    ax.plot(list_ra_PFoV, list_dec_PFoV, 'b,', alpha=0.5)
    ax.plot(list_ra_FFoV, list_dec_FFoV, 'r,', alpha=0.5)
    ax.plot(source.alpha, source.delta, 'k+')
    ax.set_xlabel('right ascension [rads]')
    ax.set_ylabel('declination [rads]')
    ax.set_xlim(min(list_ra_PFoV), max(list_ra_PFoV))
    ax.set_ylim(min(list_dec_PFoV), max(list_dec_PFoV))
    ax.grid()
    return fig, ax


# ### For sources updating: ----------------------------------------------------
def plot_errors_VS_iterations_per_source(Solver, save_path=None):
    """
    Plots the error on each astronomic parameter and objective function for each
    source.

    :param Solver: [Solver object]
    :param save_path: [string] path to the saving folder
    :return: list of figures wih the plots for each source
    """
    figs_list = []
    for source_index, s in enumerate(Solver.calc_sources):
        calc_source = Solver.calc_sources[source_index]
        real_source = Solver.real_sources[source_index]

        my_title = s.name
        my_observations = calc_source.obs_times
        source_params = np.array(calc_source.s_old)
        fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharex='all')

        fig.suptitle(my_title, fontsize=40)

        num_iters = len(calc_source.errors)
        labels = ['alpha', 'delta', 'parallax', 'mu_alpha', 'mu_delta']
        real_source.set_time(0)
        observed = [real_source.alpha, real_source.delta, real_source.parallax,
                    real_source.mu_alpha_dx, real_source.mu_delta]

        alpha_list, delta_list = ([], [])
        for t_L in my_observations:
            real_source.set_time(float(t_L))
            alpha_list.append(real_source.alpha)
            delta_list.append(real_source.delta)

        for i, x in enumerate(source_params.T):
            if i < 3:
                ax = axs[0, i]
            else:
                ax = axs[1, i-3]
            if (i < 3):
                ax.semilogy(np.abs(observed[i] - x)/const.rad_per_mas,
                            'b--.', label=labels[i])
                ax.set_ylabel('[mas]')
            elif (i == 3 or i == 4):
                ax.semilogy(np.abs(observed[i] - x)/const.rad_per_mas*const.days_per_year,
                            'b--.', label=labels[i])
                ax.set_ylabel('[mas/year]')
            else:
                raise ValueError('not a valid plot index')
            ax.grid()
            ax.set_label('labels[i]')
            ax.set_xlabel('Iterations')
            ax.legend()

        # plot evolution of the error
        ax = axs[-1, -1]
        ax.semilogy(calc_source.errors, 'b--.', label='objective function')
        ax.set_xlabel('Iterations')
        ax.grid(alpha=0.8)
        ax.legend()
        if save_path is not None:
            fig.savefig(save_path + '_errors_'+s.name+'.png')
        figs_list.append(fig)
    return figs_list


# ### End for sources updating #################################################


# old
def plot_phi_solutions(source, sat, obs_times, ti=0, tf=90, n=1000):
    styles = ['b.', 'rs']
    phi_limit = np.radians(0.5)
    eta_limit = np.radians(0.5)
    y_limit = (-phi_limit*10, phi_limit*10)
    times_total = np.linspace(ti, tf, n)
    phi_list = []
    eta_list = []
    phi_sol_list = []
    eta_sol_list = []
    for t in times_total:
        phi_value, eta_value = phi(source, sat, t)
        eta_list.append(eta_value)
        phi_list.append(phi_value)
    for t in obs_times:
        phi_value, eta_value = phi(source, sat, t)
        eta_sol_list.append(eta_value)
        phi_sol_list.append(phi_value)

    fig1 = plt.figure(1)
    plt.plot(times_total, phi_list, styles[0], label='phi path', alpha=0.5)
    plt.plot(obs_times, phi_sol_list, styles[1], label='solutions')
    plt.hlines(0, xmin=times_total[0], xmax=times_total[-1], color='g')
    plt.hlines(phi_limit, xmin=times_total[0], xmax=times_total[-1], color='g', linestyle='dotted',
               label='field of view limitation (5°)')
    plt.hlines(-phi_limit, xmin=times_total[0], xmax=times_total[-1], color='g', linestyle='dotted')
    plt.xlim(ti, tf)
    plt.ylim(y_limit)
    plt.xlabel('time [days]')
    plt.ylabel('Phi [rad]')
    plt.grid()
    plt.legend()

    fig2 = plt.figure(2)
    plt.plot(times_total, eta_list, styles[0], label='eta path')
    plt.plot(obs_times, eta_sol_list, styles[1], label='solutions')
    plt.hlines(0, xmin=times_total[0], xmax=times_total[-1], color='g')
    plt.hlines(eta_limit, xmin=times_total[0], xmax=times_total[-1], color='g', linestyle='dotted',
               label='field of view limitation (5°)')
    plt.hlines(-eta_limit, xmin=times_total[0], xmax=times_total[-1], color='g', linestyle='dotted')
    plt.xlim(ti, tf)
    plt.ylim(y_limit)
    plt.xlabel('time [days]')
    plt.ylabel('Eta[rad]')
    plt.grid()
    plt.legend()

    return fig1, fig2


# old
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


# old
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


# new
def plot_star_trajectory(source, sat, ti=0, tf=None, obs_times=[], equatorial=True,
                         dt=1, show_scanning_directions=False):
    """
    Plots star trajectory. One dot is plotted every day by default.
    :param source: source object
    :param sat: Satellite object
    :param tf: final time for which the trajectory is desired [days] from
    J2000.
    :param equatorial: [bool] if True use equatorial coordinates, if False use
    realtive coordinates
    :param dt: [float][days] ploints plotted very dt (default is one day)
    :returns: [figure] plot of the trajectory if the star
    """
    # if tf is not defined, this sets tf as the final time for satellite
    if tf is None:
        tf = sat.storage[-1][0]

    alphas, deltas = ([], [])
    alphas_sol, deltas_sol = ([], [])  # coordinates of the founds sources
    times_sol = []

    for i in np.arange(ti, tf, 1):
        alpha, delta, delta_alpha_dx_mas, delta_delta_mas = source.topocentric_angles(sat, i)
        if equatorial is True:
            alphas.append(alpha)
            deltas.append(delta)
        else:
            alphas.append(delta_alpha_dx_mas)
            deltas.append(delta_delta_mas)

    if obs_times:  # if list is not empty
        for t in obs_times:
            alpha_obs, delta_obs, delta_alpha_dx_mas, delta_delta_mas = source.topocentric_angles(sat, t)
            times_sol.append(t/const.days_per_year + 2000)  # +2000 do the fact that the reference epoch is J2000
            if equatorial is True:
                alphas_sol.append(alpha_obs)
                deltas_sol.append(delta_obs)
            else:
                alphas_sol.append(delta_alpha_dx_mas)
                deltas_sol.append(delta_delta_mas)

    # Convert all in mas:
    if equatorial is True:  # the other are already in mas!
        alphas = np.array(alphas) / const.rad_per_mas
        deltas = np.array(deltas) / const.rad_per_mas
        if obs_times:
            alphas_sol = np.array(alphas_sol) / const.rad_per_mas
            deltas_sol = np.array(deltas_sol) / const.rad_per_mas

    # Styles for the plots
    path_style, origin_style, sol_style = ('b,', 'ks', 'rs')

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    fig.suptitle(r'$\varpi={%f}$ [mas], $\mu_{{\alpha*}}={%f}$ [mas/year], $\mu_\delta={%f}$ [mas/year]'
                 % (source.parallax/const.rad_per_mas,
                    source.mu_alpha_dx / const.rad_per_mas / const.days_per_year,
                    source.mu_delta / const.rad_per_mas / const.days_per_year),
                 fontsize=20)

    times = np.linspace(2000, 2000 + tf/const.days_per_year, len(alphas))
    # cmaps: 'jet', 'winter', 'viridis', 'plasma'
    ax.scatter(alphas, deltas, c=times, marker='.', s=(72./fig.dpi)**2, cmap='jet',
               alpha=0.5, label=r'%s path' % (source.name), lw=2)

    ax.plot(alphas[0], deltas[0], origin_style, label='origin')  # plot origin
    ax.plot(alphas_sol, deltas_sol, sol_style, label='solutions')

    ax.set_xlim(np.min(alphas), np.max(alphas))
    ax.set_ylim(np.min(deltas), np.max(deltas))
    if equatorial is True:
        ax.set_xlabel(r'$\alpha*$ [mas]')
        ax.set_ylabel(r'$\delta$ [mas]')
    else:
        ax.axhline(y=0, c='gray', lw=1)
        ax.axvline(x=0, c='gray', lw=1)
        ax.set_xlabel(r'$\Delta\alpha*$ [mas]')
        ax.set_ylabel(r'$\Delta\delta$ [mas]')

    ax.legend()
    return fig


# old
def plot_3D_scanner_pos(sat, axis, ti, tf, n_points=1000, elevation=10, azimuth=10, figsize=(12, 12)):
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
    if axis not in ['X', 'Z']:
        raise ValueError("Axis can be either 'X' or 'Z'.")

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

    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')

    ax.plot(listx, listy, listz, style_, label=label_)
    ax.plot([0], [0], [0], 'kx', label='origin')
    ax.legend()
    ax.set_xlabel('L')
    ax.set_ylabel('M')
    ax.set_zlabel('N')
    ax.azim = azimuth
    ax.elev = elevation
    return fig  # plt.show()


# this function should not be used
def plot_ICRS_coordinates_versus_time(source, sat, obs_times=[]):
    """
    :param source: [Source]
    :param sat: [Satellite]
    :param obs_times: [list] observed times
    :return: plot of the righ ascension and declination in CoMRS
    """
    time_total = sat.storage[-1][0]

    alphas, deltas = ([], [])
    alphas_sol, deltas_sol = ([], [])
    times_sol = []

    for i in np.arange(0, time_total, 1):
        alpha, delta, delta_alpha_dx_mas, delta_delta_mas = source.topocentric_angles(sat, i)
        alphas.append(delta_alpha_dx_mas)
        deltas.append(delta_delta_mas)

    for t in obs_times:
        times_sol.append(t/const.days_per_year + 2000)  # +2000 do the fact that the reference epoch is J2000
        alpha_obs, delta_obs, delta_alpha_dx_mas, delta_delta_mas = source.topocentric_angles(sat, t)
        alphas_sol.append(delta_alpha_dx_mas)
        deltas_sol.append(delta_delta_mas)

    n = len(alphas)
    times = np.linspace(2000, 2000 + time_total/const.days_per_year, n)

    # Styles for the plots
    path_style = 'b,'
    origin_style = 'ks'
    sol_style = 'rs'

    fig = plt.figure(figsize=(16, 9))

    # Top subplot
    ax1dra = fig.add_subplot(211)
    ax1dra.plot(times, alphas, path_style)
    ax1dra.plot(times[0], alphas[0], origin_style, label='origin')
    ax1dra.plot(times_sol, alphas_sol, sol_style, label='solutions')
    ax1dra.axhline(y=0, c='gray', lw=1)
    # ax1dra.set_xlabel(r'Time [yr]')
    ax1dra.set_ylabel(r'$\Delta\alpha*$ [mas]')

    # Top left subplot
    ax1ddec = fig.add_subplot(212, sharex=ax1dra)
    ax1ddec.axhline(y=0, c='gray', lw=1)
    ax1ddec.plot(times, deltas, path_style)
    ax1ddec.plot(times[0], deltas[0], origin_style, label='origin')
    ax1ddec.plot(times_sol, deltas_sol, sol_style, label='solutions')
    ax1ddec.set_xlabel(r'Time [yr]')
    ax1ddec.set_ylabel(r'$\Delta\delta$ [mas]')

    # plt.tight_layout()
    plt.show()
