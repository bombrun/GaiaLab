# -*- coding: utf-8 -*-
import numpy as np
from quaternion import Quaternion


def to_quaternion(vector):
    """
    converts vector to quaternion with first component set to zero.
    :param vector: 3D np.array
    :return: Quaternion (0, vector*)
    """
    return Quaternion(0, vector[0], vector[1], vector[2])


def alpha_delta(vector):

    alpha = np.arctan2(vector[1], vector[0])
    delta = np.arctan2(vector[2], np.sqrt(vector[1]**2 + vector[0]**2))  
    return alpha, delta


def xyz(azimuth, altitude):

    x = np.cos(azimuth)*np.cos(altitude)
    y = np.sin(azimuth)*np.cos(altitude)
    z = np.sin(altitude)
    return np.array([x, y, z])


def ljk(epsilon):
    """
    Calculates ecliptic triad vectors with respect to BCRS-frame.
    (Lindegren, SAG-LL-35, Eq.1)

    :param epsilon: obliquity of the equator.
    :return: np.array, np.array, np.array
    """
    l = np.array([1,0,0])
    j = np.array([0, np.cos(epsilon), np.sin(epsilon)])
    k = np.array([0, -np.sin(epsilon), np.cos(epsilon)])
    return l, j, k


def rotation_to_quat(vector, angle):
    """
    Calculates quaternion equivalent to rotation about (vector) by an (angle).
    :param vector:
    :param angle:
    :return:
    """
    vector = vector / np.linalg.norm(vector)
    t = np.cos(angle/2.)
    x = np.sin(angle/2.) * vector[0]
    y = np.sin(angle/2.) * vector[1]
    z = np.sin(angle/2.) * vector[2]

    return Quaternion(t, x, y, z)


def bcrs(attitude, vector):
    '''
    Changes coordinates of a vector in BCRS to SRS frame.
    '''
    q_vector_bcrs = to_quaternion(vector)
    q_vector_srs = attitude * q_vector_bcrs * attitude.conjugate()

    return q_vector_srs.to_vector()


def srs(attitude, vector):
    '''
    Changes coordinates of a vector in SRS to BCRS frame.
    '''
    q_vector_srs = to_quaternion(vector)
    q_vector_bcrs = attitude.conjugate() * q_vector_srs * attitude  
    return q_vector_bcrs.to_vector()