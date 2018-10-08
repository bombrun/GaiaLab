"""
agis_helpers.py
functions that uses the classes scanner, source, satellite but don't belong to a
given file yet
author: LucaZampieri
"""

# # Imports
# Global imports
import numpy as np
# Local imports
import constants as const
import helpers as helpers
import frame_transformations as ft
from quaternion import Quaternion
from source import Source
from satellite import Satellite
from scanner import Scanner


def compute_field_angles(source, sat, t):
    """
    Return field angles according to Lindegren eq. 12
    eta: along-scan field angle
    """
    Gamma_c = 0  # angle between the two scanners # TODO: implement gamma_c
    Cu = source.unit_topocentric_function(sat, t)  # u in CoMRS frame
    Su = ft.lmn_to_xyz(sat.func_attitude(t), Cu)  # u in SRS frame
    Su_x = Su[0]
    Su_y = Su[1]
    Su_z = Su[2]
    phi = np.arctan2(Su_y, Su_x)
    if phi >= np.pi or phi < -np.pi:
        raise ValueError('phi should be -pi <= phi < pi, instead it is: {}'.format(phi))
    xi = np.arctan2(Su_z, np.sqrt(Su_x**2+Su_y**2))
    field_index = np.sign(phi)
    eta = phi - field_index * Gamma_c / 2
    return eta, xi


def compute_mnu(eta, xi):
    """
    return column vectors of the S'[m_l, n_l, u_l] matrix
    """
    phi = eta  # # TODO: implement the correct version (phi != eta)
    # S_mnu = np.zeros((3,3))
    m_l = np.array([-np.sin(phi), np.cos(phi), 0])
    n_l = np.array([-np.sin(xi)*np.cos(phi), np.sin(xi)*np.sin(phi), np.cos(xi)])
    u_l = np.array([np.cos(xi)*np.cos(phi), np.cos(xi)*np.sin(phi), np.sin(xi)])
    return np.array([m_l, n_l, u_l])


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
    direction_lmn = u_lmn_unit - sat.func_x_axis_lmn(t)
    direction_xyz = ft.lmn_to_xyz(sat.func_attitude(t), direction_lmn)
    phi = np.arcsin(direction_xyz[1])
    eta = np.arcsin(direction_xyz[2])
    # phi = np.arctan2(direction_xyz[1], direction_xyz[0])
    # xi = np.arctan2(direction_xyz[2], np.sqrt(direction_xyz[0] ** 2 + direction_xyz[1] ** 2))
    return phi, eta
