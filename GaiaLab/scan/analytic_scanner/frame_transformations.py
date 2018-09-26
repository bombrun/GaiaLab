# -*- coding: utf-8 -*-

"""
Created in jun 2018

@author: mdelvallevaro

modified by: LucaZampieri 2018

"""

import numpy as np
from quaternion import Quaternion


def vector_to_quaternion(vector):
    """
    converts vector to quaternion with first component set to zero.
    :param vector: 3D np.array
    :return: Quaternion (0, vector*)
    """
    return Quaternion(0, vector[0], vector[1], vector[2])


def vector_to_polar(vector):
    """
    Convert carthesian coordinates of a vector into its corresponding polar coordinates
    :param vector: [pc]
    :return: [rad][rad][pc] alpha, delta, radius
    """
    radius = np.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    alpha = np.arctan2(vector[1], vector[0]) % (2*np.pi)
    delta = np.arcsin(vector[2]/radius)
    return alpha, delta, radius


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
    :return: [parsec](x, y, z)array in parsecs.
    """
    parallax = parallax/1000  # from mas to arcsec

    x = (1/parallax)*np.cos(delta)*np.cos(alpha)
    y = (1/parallax)*np.cos(delta)*np.sin(alpha)
    z = (1/parallax)*np.sin(delta)

    return np.array([x, y, z])


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


def rotation_to_quat(vector, angle):
    """
    Calculates quaternion equivalent to rotation about (vector) by an (angle).
    :param vector:  [np.array]
    :param angle: [deg]
    :return equivalent quaternion:
    """
    vector = vector / np.linalg.norm(vector)
    t = np.cos(angle/2.)
    x = np.sin(angle/2.) * vector[0]
    y = np.sin(angle/2.) * vector[1]
    z = np.sin(angle/2.) * vector[2]

    return Quaternion(t, x, y, z)


def xyz_to_lmn(attitude, vector):
    """
    Go from the rotating (xyz) frame to the non-rotating (lmn) frame

    Info: The attitude Qauaternion q(t) gives the rotation from (lmn) to (xyz)
        (lmn) being the CoMRS (C), and (xyz) the SRS (S). The relation between
        the two frames is given by:
            {C'v,0} = q {S'v,0} q^-1      for an any vector v

    :param attitude: Quaternion object
    :param vector: array of 3D
    :return: the coordinates in LMN-frame of the input vector.
    """
    q_vector_xyz = vector_to_quaternion(vector)
    q_vector_lmn = attitude * q_vector_xyz * attitude.conjugate()
    return q_vector_lmn.to_vector()


def lmn_to_xyz(attitude, vector):
    """
    Goes from the non-rotating (lmn) frame to the rotating (xyz) frame

    Info: The attitude Qauaternion q(t) gives the rotation from (lmn) to (xyz)
        (lmn) being the CoMRS (C), and (xyz) the SRS (S). The relation between
        the two frames is given by:
            {S'v,0} = q^-1 {C'v,0} q      for an any vector v

    :param attitude: Quaternion object
    :param vector: array of 3D
    :return: the coordinates in XYZ-frame of the input vector.
    """
    q_vector_lmn = vector_to_quaternion(vector)
    q_vector_xyz = attitude.conjugate() * q_vector_lmn * attitude
    return q_vector_xyz.to_vector()


def parsecs_to_au(vector):
    return vector * 206265
