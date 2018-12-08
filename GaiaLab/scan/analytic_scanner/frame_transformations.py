# -*- coding: utf-8 -*-
"""
File frame_transformations.py
Contains functions that for frame transformations and rotations

:Authors: mdelvallevaro, LucaZampieri (2018) modified

:notes:
    In this file, when there is a reference, unless explicitly stated otherwise,
    it refers to Lindegren main article:
    "The astronometric core solution for the Gaia mission - overview of models,
    algorithms, and software implementation" by L. Lindegren, U. Lammer, D. Hobbs,
    W. O'Mullane, U. Bastian, and J.Hernandez
    The reference is usually made in the following way: Ref. Paper eq. [1]

"""

import numpy as np
import quaternion


def zero_to_two_pi_to_minus_pi_pi(angle):
    """
    Tranforms an angle in range [0-2*pi] to range [-pi, pi]
    :param angle: [rad] angle or array of angles in [0-2*pi] format
    :returns: angles in the [-pi, pi] format
    """
    indices_to_modify = np.where(angle > np.pi)
    angle[indices_to_modify] = angle[indices_to_modify] - 2*np.pi
    return angle


def rotate_by_angle(vector, angle):
    pass
    """quaternion = Quaternion(vector=vector, angle=angle)
    rotated_vector = rotate_by_quaternion(quaternion, vector)
    return rotated_vector"""


def vector_to_polar(vector):
    """
    Convert carthesian coordinates of a vector into its corresponding polar coordinates
    :param vector: [pc]
    :return: [rad][rad][pc] alpha, delta, radius
    """
    radius = np.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    alpha = np.arctan2(vector[1], vector[0]) % (2*np.pi)
    delta = np.arcsin(vector[2]/radius)
    dist_xy = np.sqrt(vector[0]**2+vector[1]**2)
    delta = np.arctan2(vector[2],  dist_xy) % (2*np.pi)
    return alpha, delta, radius


def vector_to_alpha_delta(vector):
    """
    Ref. Paper eq. [96]
    Convert carthesian coordinates of a vector into its corresponding polar
    coordinates (0 - 2*pi)
    :param vector: [whatever] X,Y,Z coordinates in CoMRS frame (non-rotating)
    :return: [rad][rad] alpha, delta --> between 0 and 2*pi (in ICRS coordinates)
    """
    alpha = np.arctan2(vector[1], vector[0]) % (2*np.pi)
    dist_xy = np.sqrt(vector[0]**2+vector[1]**2)
    delta = np.arctan2(vector[2],  dist_xy) % (2*np.pi)
    return alpha, delta


def polar_to_direction(alpha, delta):
    """
    Convert polar angles to unit direction vector
    :param alpha: [rad]
    :param delta: [rad]
    :returns: 3D np.array unit vector
    """
    x = np.cos(alpha)*np.cos(delta)
    y = np.sin(alpha)*np.cos(delta)
    z = np.sin(delta)
    return np.array([x, y, z])


def adp_to_cartesian(alpha, delta, parallax):
    """
    Convert coordinates from (alpha, delta, parallax) format into the (x, y, z)
    format.
    :param azimuth: [rad]
    :param altitude: [rad]
    :param parallax: [mas]
    :return: [parsec](x, y, z) array in parsecs.
    """
    parallax = 1  # parallax/1000  # from mas to arcsec
    # parallax = parallax/const.rad_per_arcsec
    # WARNING: but why parallax??
    x = (1/parallax)*np.cos(delta)*np.cos(alpha)
    y = (1/parallax)*np.cos(delta)*np.sin(alpha)
    z = (1/parallax)*np.sin(delta)

    return np.array([x, y, z])


def vector_to_adp(vector, tolerance=1e-6):
    """
    :return: alpha, delta in radians
    """
    x, y, z = vector[:]
    delta = np.arcsin(z)
    alpha_1 = np.arccos(x/np.cos(delta))
    alpha_2 = np.arccos(x/np.cos(delta))
    diff_a1_a2 = alpha_1 - alpha_2
    mean_alpha = (alpha_1 + alpha_2) / 2
    relative_error = diff_a1_a2/mean_alpha
    if relative_error > tolerance:
        raise ValueError('relative difference in alpha of {} is too big'.format(relative_error))
    return mean_alpha, delta


def compute_ljk(epsilon):
    """
    Calculates ecliptic triad vectors with respect to BCRS-frame.
    (Lindegren, SAG-LL-35, Eq.1)

    :param epsilon: obliquity of the equator.
    :return: np.array, np.array, np.array

    """
    L = np.array([1, 0, 0])
    j = np.array([0, np.cos(epsilon), np.sin(epsilon)])
    k = np.array([0, -np.sin(epsilon), np.cos(epsilon)])
    return L, j, k


def compute_pqr(alpha, delta):
    """
    Ref. Paper eq. [5]
    .. note::
        Can be used also with numpy arrays

    :param alpha: [rad] astronomic parameter alpha
    :param delta: [rad] astronomic parameter alpha
    :returns: p, q, r
    """
    p = np.array([-np.sin(alpha), np.cos(alpha), 0])
    q = np.array([-np.sin(delta)*np.cos(alpha), -np.sin(delta)*np.sin(alpha),
                  np.cos(delta)])
    r = np.array([np.cos(delta)*np.cos(alpha), np.cos(delta)*np.sin(alpha),
                  np.sin(delta)])

    return p, q, r


def rotate_by_quaternion(quaternion, vector):
    """
    Ref. Paper eq. [9]
    rotate vector by quaternion
    """
    q_vector = vector_to_quat(vector)
    q_rotated_vector = quaternion * q_vector * quaternion.inverse()
    return quat_to_vector(q_rotated_vector)


def xyz_to_lmn_old(attitude, vector):
    """
    Ref. Paper eq. [9]
    Go from the rotating (xyz) frame to the non-rotating (lmn) frame

    Info:
        The attitude Qauaternion q(t) gives the rotation from (lmn) to (xyz)
        (lmn) being the CoMRS (C), and (xyz) the SRS (S). The relation between
        the two frames is given by: {C'v,0} = q {S'v,0} q^-1 for an any vector v

    :param attitude: Quaternion object
    :param vector: array of 3D
    :return: the coordinates in LMN-frame of the input vector.
    """
    pass


def lmn_to_xyz_old(attitude, vector):
    """
    Ref. Paper eq. [9]
    Goes from the non-rotating (lmn) frame to the rotating (xyz) frame

    Info: The attitude Qauaternion q(t) gives the rotation from (lmn) to (xyz)
        (lmn) being the CoMRS (C), and (xyz) the SRS (S). The relation between
        the two frames is given by: {S'v,0} = q^-1 {C'v,0} q for an any vector v

    :param attitude: Quaternion object
    :param vector: array of 3D
    :return: the coordinates in XYZ-frame of the input vector.
    """
    pass


# ### For mobble quaternion
def quat_to_vector(quat):
    return quaternion.as_float_array(quat)[1:]


def vector_to_quat(vector):
    return np.quaternion(0, vector[0], vector[1], vector[2])


def xyz_to_lmn(attitude, vector):
    q_vector_xyz = vector_to_quat(vector)
    q_vector_lmn = attitude * q_vector_xyz * attitude.inverse()
    return quat_to_vector(q_vector_lmn)


def lmn_to_xyz(attitude, vector):
    q_vector_lmn = vector_to_quat(vector)
    q_vector_xyz = attitude.inverse() * q_vector_lmn * attitude
    return quat_to_vector(q_vector_xyz)
