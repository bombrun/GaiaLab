# -*- coding: utf-8 -*-


import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from quaternion import Quaternion
import frame_transformations as ft

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

    ax.plot(x_listx, x_listy, x_listz, 'bo', label='X vector rotation')
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

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.show()

def plot_general_3d(lista_3d):
    x = [i[0] for i in lista_3d]
    y  = [i[1] for i in lista_3d]
    z = [i[2] for i in lista_3d]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x,y,z, 'bo--')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.show()

def plot_attitude_mag(att, ti, tf, n_points=100):

    times= np.linspace(ti, tf, n_points)
    attitudes = [att.get_attitude(t) for t in times]

    norm_list = [np.sqrt(obj.w**2 + obj.x**2 + obj.y**2 + obj.z**2) for obj in attitudes]

    plt.figure()
    plt.plot(times,norm_list, 'bo')

    plt.show()


def plot_attitude(att, ti, tf, n_points):
    '''
    Args
    ______
    satellite: object to be calculated and updated.
    dt: step per day, i.e fraction of a day.
    n0: initial day
    nf: final days


    Returns
    ________
    Plot of the 4 components of the attitude of the satellite.
    attitude = (t, x, y, z)
    Each graph plots time in days versus each component evolution wrt time.
    '''
    times= np.linspace(ti, tf, n_points)
    attitudes = [att.func_attitude(t) for t in times]

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

    x_list = [star.coor[0] for star in sky.elements]
    y_list = [star.coor[1] for star in sky.elements]
    z_list = [star.coor[2] for star in sky.elements]

    mpl.rcParams['legend.fontsize'] = 10

    fig2 = plt.figure()
    ax = fig2.gca(projection='3d')

    ax.plot(x_list, y_list, z_list, 'b*', label='observation measurements')
    ax.legend()
    ax.set_xlabel('l')
    ax.set_ylabel('m')
    ax.set_zlabel('n')
    plt.show()

def plot_observations_spread(source, satellite, scan):
    alphas_obs = []
    deltas_obs = []

    #data info
    for t in scan.obs_times:
        alpha, delta, radius = ft.to_polar(satellite.func_x_axis_lmn(t))
        alphas_obs.append(alpha)
        deltas_obs.append(delta)
    #data arrows
    zalphas = []
    zdeltas = []
    for t in scan.obs_times:
        z_alpha, z_delta, z_radio  = ft.to_polar(satellite.func_z_axis_lmn(t))
        zalphas.append(z_alpha)
        zdeltas.append(z_delta)


    plt.figure()
    plt.plot(alphas_obs, deltas_obs, 'ro')
    plt.quiver(alphas_obs, deltas_obs, zalphas, zdeltas) #headaxislength = 0.1, normalize = True)
    plt.xlabel('alpha [rad]')
    plt.ylabel('delta [rad]')
    plt.show()

def plot_directions (satellite, scan):
    points = []
    for t in scan.obs_times:
        vector = satellite.func_x_axis_lmn(t)
        points.append(vector)
    xvalues = [i[0] for i in arrows]
    yvalues = [i[1] for i in arrows]

    plt.figure()
    plt.quiver(xvalues, yvalues, length = 0.1, normalize = True)
    plt.show()

def plot_star_trajectory(source, satellite, scan, t_total):
    mastorad = 2 * np.pi / (1000 * 360 * 3600)
    alphas = []
    deltas = []

    for i in np.arange(0, t_total, 1):
        delta_alpha, delta_delta  =  source.topocentric_angles(satellite, i)
        alphas.append(delta_alpha)
        deltas.append(delta_delta)

    alphas_obs = []
    deltas_obs = []
    #intercept stars
    for t in scan.obs_times:
        delta_alpha_obs, delta_delta_obs = source.topocentric_angles(satellite, t)
        alphas_obs.append(delta_alpha_obs)
        deltas_obs.append(delta_delta_obs)

    alphas_sat = []
    deltas_sat = []
    #from satellite
    for t in scan.obs_times:
        alpha, delta, radius = ft.to_polar(satellite.func_x_axis_lmn(t))
        alphas_sat.append(alpha/mastorad)
        deltas_sat.append(delta/mastorad)

    #timeline
    n = len(alphas)
    times = np.linspace(2000, 2000 + t_total/365, n)
    times_observations = [2000 + t/365 for t in scan.obs_times]

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(121)

    ax.plot(alphas, deltas,'b-',
            label=r'%s path' %(source.name), lw=2)
    ax.plot(alphas_obs, deltas_obs, 'r*')
    ax.plot(alphas_sat, deltas_sat, 'g*')
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
    ax1dra.plot(times_observations,alphas_obs, 'r*')
    ax1dra.plot(times_observations, alphas_sat, 'g*')
    ax1dra.set_ylabel(r'$\Delta\alpha*$ [mas]')

    ax1ddec = fig.add_subplot(224)
    ax1ddec.axhline(y=0, c='gray', lw=1
    ax1ddec.plot(times, deltas, 'b-')
    ax1ddec.plot(times_observations, deltas_obs, 'r*')
    ax1ddec.plot(times_observations, deltas_sat, 'g*')
    ax1ddec.set_xlabel(r'Time [yr]')
    ax1ddec.set_ylabel(r'$\Delta\delta$ [mas]')

    plt.tight_layout()
    plt.show()







			




