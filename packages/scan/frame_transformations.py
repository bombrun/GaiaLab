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


def alpha_delta_radius(vector):

    solar_radius = np.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    alpha = np.arctan2(vector[1], vector[0])
    delta = np.arcsin(vector[2]/solar_radius)
    return alpha, delta, solar_radius


def cartesian_coord(azimuth, altitude, parallax = 1):

    x = (1/np.tan(parallax))*np.cos(azimuth)*np.cos(altitude)
    y = (1/np.tan(parallax))*np.sin(azimuth)*np.cos(altitude)
    z = (1/np.tan(parallax))*np.sin(altitude)

    return np.array([x, y, z])


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

def lmn_srs(vector_lmn, epsilon = np.radians(23.27)):
    l = np.array([1, 0, 0])
    j = np.array([0, np.cos(epsilon), -np.sin(epsilon)])
    k = np.array([0, np.sin(epsilon), np.cos(epsilon)])

    A = np.vstack([l, j, k])
    A_matrix = A.reshape(3, 3)

    vector_srs = np.dot(A_matrix, vector_lmn)
    return vector_srs

def srs_lmn(vector_srs, epsilon=np.radians(23.27)):
    l = np.array([1, 0, 0])
    j = np.array([0, np.cos(epsilon), np.sin(epsilon)])
    k = np.array([0, -np.sin(epsilon), np.cos(epsilon)])

    A = np.vstack([l, j, k])
    A_matrix = A.reshape(3, 3)

    vector_lmn = np.dot(A_matrix, vector_srs)
    return vector_lmn

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


def xyz(attitude, vector):
    q_vector_srs = to_quaternion(vector)
    q_vector_xyz = attitude * q_vector_srs * attitude.conjugate()
    return q_vector_xyz.to_vector()

def srs(attitude, vector):
    q_vector_xyz = to_quaternion(vector)
    q_vector_srs = attitude.conjugate() * q_vector_xyz * attitude
    return q_vector_srs.to_vector()


