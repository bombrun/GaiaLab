# -*- coding: utf-8 -*-
"""
File frame_transformations.py
Contains functions that for frame transformations and rotations

:Authors: mdelvallevaro, LucaZampieri (2018)

.. note:: In this file, when there is a reference, unless explicitly stated otherwise,
    it refers to Lindegren main article:
    "The astronometric core solution for the Gaia mission - overview of models,
    algorithms, and software implementation" by L. Lindegren, U. Lammer, D. Hobbs,
    W. O'Mullane, U. Bastian, and J.Hernandez
    The reference is usually made in the following way: Ref. Paper [LUDW2011]_ eq. [1]

"""

import numpy as np
import quaternion


def zero_to_two_pi_to_minus_pi_pi(angle, unit='radians'):
    """
    Tranforms an angle in range [0-2*pi] to range [-pi, pi] by substracting 2pi
    to any angle greater than pi.

    Info: Can be used with numpy arrays

    :param angle: [rad] angle or array of angles in [0-2*pi] format
    :param unit: [str] specify if the input data is in radians or degrees
    :returns: angle in the [-pi, pi] format
    """
    if unit == 'radians':
        indices_to_modify = np.where(angle > np.pi)
        angle[indices_to_modify] = angle[indices_to_modify] - 2*np.pi
    elif unit == 'degrees':
        indices_to_modify = np.where(angle > 180)
        angle[indices_to_modify] = angle[indices_to_modify] - 360
    else:
        raise ValueError('Not a valid *unit* value. Can be degrees or radians')
    return angle


def transform_twoPi_into_halfPi(deltas):
    """
    Tranforms an angle in range [0-2*pi] to range [-pi/2, pi/2] by substracting 2pi
    to any angle greater than pi.


    .. warning:: The input angles have to be defined between [0,pi/2] and
        [3pi/2, 2pi]

    :param delta: input angles
    :return: modified angles
    """
    deltas = np.array(deltas)
    to_modify_indices = np.where(deltas > np.pi)[0]
    deltas[to_modify_indices] -= 2*np.pi
    return deltas


def vector_to_alpha_delta(vector, two_pi=False):
    """
    Ref. Paper [LUDW2011]_ eq. [96]
    Convert carthesian coordinates of a vector into its corresponding polar
    coordinates (0 - 2*pi)

    :param vector: [array] X,Y,Z coordinates in CoMRS frame (non-rotating)
    :param two_pi: [bool] if True return delta in [0,2pi] instead of [-pi/2,pi/2]
    :return: [rad][rad] alpha, delta
    """
    radius = np.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    delta = np.arcsin(vector[2]/radius)  # gives delta in [-pi/2, pi/2]
    alpha = np.arctan2(vector[1], vector[0]) % (2*np.pi)
    if two_pi is True:
        dist_xy = np.sqrt(vector[0]**2+vector[1]**2)
        delta = np.arctan2(vector[2],  dist_xy) % (2*np.pi)  # gives delta in [0, 2pi]
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

    parallax = parallax / const.rad_per_arcsec  # from rad to arcsec
    x = (1/parallax)*np.cos(delta)*np.cos(alpha)
    y = (1/parallax)*np.cos(delta)*np.sin(alpha)
    z = (1/parallax)*np.sin(delta)
    return np.array([x, y, z])


def compute_ljk(epsilon):
    """
    | Ref. [Lind2001]_ (Lindegren, SAG-LL-35, Eq.1)
    | Calculates ecliptic triad vectors with respect to BCRS-frame.

    :param epsilon: obliquity of the equator.
    :return:
        np.array, np.array, np.array of the ecliptic triad:
            * **l**: is a unit vector toward (alpha, delta) = (0,0)`)
            * **n**: is a unit vector towards delta = 90°
            * **m** = **n** x **l**
    """
    L = np.array([1, 0, 0])
    j = np.array([0, np.cos(epsilon), np.sin(epsilon)])
    k = np.array([0, -np.sin(epsilon), np.cos(epsilon)])
    return L, j, k


def compute_pqr(alpha, delta):
    """
    | Ref. Paper [LUDW2011]_ eq. [5]
    | **Can be used also with numpy arrays**
    | Computes the p, q, r parameters


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
    Ref. Paper [LUDW2011]_ eq. [9]
    rotate vector by quaternion
    """
    q_vector = vector_to_quat(vector)
    q_rotated_vector = quaternion * q_vector * quaternion.inverse()
    return quat_to_vector(q_rotated_vector)


def quat_to_vector(quat):
    """
    :param quat: [quaternion] Quaternion to transform into vector
    :return: 3D array made with x,y,z components of the quaternion
    """
    return quaternion.as_float_array(quat)[1:]


def vector_to_quat(vector):
    """
    Transform vector to quaternion by setting x,y,z components of the quaternion
    with x,y,z components of the vector.

    :param vector: vector to transform to quaternion
    :return: quaternion created from vector
    """
    return np.quaternion(0, vector[0], vector[1], vector[2])


def xyz_to_lmn(attitude, vector):
    """
    Ref. Paper [LUDW2011]_ eq. [9]
    Go from the rotating (xyz) SRS frame to the non-rotating (lmn) CoMRS frame

    .. note:: The attitude Quaternion q(t) gives the rotation from (lmn) to (xyz)
        (lmn) being the CoMRS (C), and (xyz) the SRS (S). The relation between the
        two frames is given by: {C'v,0} = q {S'v,0} q^-1 for an any vector v

    :param attitude: Quaternion object
    :param vector: array of 3D
    :return: the coordinates in LMN-frame of the input vector.
    """
    q_vector_xyz = vector_to_quat(vector)
    q_vector_lmn = attitude * q_vector_xyz * attitude.inverse()
    return quat_to_vector(q_vector_lmn)


def lmn_to_xyz(attitude, vector):
    """
    Ref. Paper [LUDW2011]_ eq. [9]
    Goes from the non-rotating (lmn) CoMRS frame to the rotating (xyz) SRS frame

    .. note:: The attitude Quaternion q(t) gives the rotation from (lmn) to (xyz)
        (lmn) being the CoMRS (C), and (xyz) the SRS (S). The relation between
        the two frames is given by: {S'v,0} = q^-1 {C'v,0} q for an any vector v

    :param attitude: Quaternion object
    :param vector: array of 3D
    :return: the coordinates in XYZ-frame of the input vector.
    """
    q_vector_lmn = vector_to_quat(vector)
    q_vector_xyz = attitude.inverse() * q_vector_lmn * attitude
    return quat_to_vector(q_vector_xyz)
