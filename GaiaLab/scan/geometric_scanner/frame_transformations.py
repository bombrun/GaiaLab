# -*- coding: utf-8 -*-
import numpy as np
from .quaternion import Quaternion


def to_quaternion(vector):
    """
    converts vector to quaternion with first component set to zero.
    :param vector: 3D np.array
    :return: Quaternion (0, vector*)
    """
    return Quaternion(0, vector[0], vector[1], vector[2])


def to_polar(vector):
    """

    :param vector: [pc]
    :return: [rad][rad][pc]
    """
    radius = np.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    alpha = np.arctan2(vector[1], vector[0])
    delta = np.arcsin(vector[2]/radius)
    return alpha, delta, radius


def to_direction(alpha, delta):
    x = np.cos(alpha)*np.cos(delta)
    y = np.sin(alpha)*np.cos(delta)
    z = np.sin(delta)

    return np.array([x, y, z])


def to_cartesian(alpha, delta, parallax):
    """

    :param azimuth: rad
    :param altitude:rad
    :param parallax: mas
    :return: array in parsecs.
    """
    parallax = parallax/1000 #from mas to arcsec

    x = (1/parallax)*np.cos(alpha)*np.cos(delta)
    y = (1/parallax)*np.sin(alpha)*np.cos(delta)
    z = (1/parallax)*np.sin(delta)

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


def pqr(alpha, delta):
    p = np.array([-np.sin(alpha), np.cos(alpha), 0])
    q = np.array([-np.sin(delta)*np.cos(alpha), -np.sin(delta)*np.sin(alpha), np.cos(delta)])
    r = np.array([np.cos(delta)*np.cos(alpha), np.cos(delta)*np.sin(alpha), np.sin(delta)])

    return p, q, r

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

def to_lmn(attitude, vector):
    q_vector_xyz = to_quaternion(vector)
    q_vector_lmn = attitude * q_vector_xyz * attitude.conjugate()
    return q_vector_lmn.to_vector()

def to_xyz(attitude, vector):
    q_vector_lmn = to_quaternion(vector)
    q_vector_xyz = attitude.conjugate() * q_vector_lmn * attitude
    return q_vector_xyz.to_vector()

def parsecs_to_au(vector):
    return vector * 206265
