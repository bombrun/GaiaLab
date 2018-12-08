# -*- coding: utf-8 -*-
"""
File helpers.py

Helper functions for the analytic scanner

Contains: (at least)
    - compute_intersection
    - compute_angle

:Author: LucaZampieri (2018)
"""

# # Imports
import numpy as np
import matplotlib.pylab as plt
import scipy.sparse as sps


def make_symmetric(sparse_matrix):
    rows, cols = sparse_matrix.nonzero()
    sparse_matrix[cols, rows] = sparse_matrix[rows, cols]

    return sparse_matrix


def get_sparse_diagonal_matrix_from_half_band(half_band):
    """it assumes we have the upper hal of the band, i.e. we some zeros at the
    bottom of band for column index greater than 0"""
    half_band = np.array(half_band)
    N = half_band.shape[0]
    for i in range(0, N//4, 1):
        half_band[i*4+1, :] = np.append(half_band[i*4+1, 1:], [0], axis=0)
        half_band[i*4+2, :] = np.append(half_band[i*4+2, 2:], [0, 0], axis=0)
        half_band[i*4+3, :] = np.append(half_band[i*4+3, 3:], [0, 0, 0], axis=0)
    half_band_width = half_band.shape[1]
    diags = []
    for i in range(0, half_band_width):
        diags.append(half_band[:, i])

    data = np.array(diags)
    # print(diags)
    offsets = np.array(range(0, -half_band_width, -1))
    # print(diags)
    Low = sps.spdiags(data, offsets, N, N)  # lower triangular
    banded_matrix = Low + Low.T - sps.diags(Low.diagonal())  # symmetrize matrix
    return banded_matrix


def normalize(v, tol=1e-10):
    """return normalized version of v"""
    norm = np.linalg.norm(v)
    if norm == 0:
        # raise ValueError('vector norm close to 0, cannot normalise vector')
        return v
    return v/norm


def check_symmetry(a, tol=1e-12):
    """ Check the symmetry of array a. True if symmetric up to tolerance"""
    return np.allclose(a, a.T, atol=tol)


def get_lists_intersection(list1, list2):
    return list(set(list1) & set(list2))


def symmetrize_triangular_matrix(a):
    """ Symmetrize an already triangular matrix (lower or upper)
    :param a: upper ot lower triangular matrix
    :returns: corresponding symmetric matrix"""
    return a + a.T - numpy.diag(a.diagonal())


def sec(x):
    """Stable if x close to 0"""
    return 1/np.cos(x)


def get_rotation_matrix(v1, v2):
    """
    Get the rotation matrix necessary to go from v1 to v2
    :param vi: 3D vector as np.array
    To rotate vector v1 into v2 then do r@v1
    """
    v1 = v1.reshape(3, 1)  # reshapes as vectors
    v2 = v2.reshape(3, 1)
    a, b = (v1 / np.linalg.norm(v1)).reshape(3), (v2 / np.linalg.norm(v2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.identity(3) + k + k@k * ((1 - c)/(s**2))  # Euler Roriguez formulae
    return R


def get_rotation_vector_and_angle(v1, v2):
    v1 = v1/np.linalg.norm(v1)  # # NOTE: it should be useless to normalize
    v2 = v2/np.linalg.norm(v2)
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    vector = np.cross(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))
    vector = vector / np.linalg.norm(vector)
    return vector, angle


# Only need numpy as np
def compute_angle(v1, v2):
    """
    Computes the angle between two Vectors
    :param vi: vector between which you want to compute the angle for each i=1:2
    :returns: [float] [deg] angle between the vectors
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def rescaled_direction(vector, length):
    unit_vector = vector/np.linalg.norm(vector)
    v = np.multiply(unit_vector, length)
    return v


# Only need numpy as np
def compute_intersection(x1, y1, x2, y2, x3, y3, x4, y4, segment=True):
    """
    Return intersection of two lines (or segments) if it exists, raise an error otherwise.
    :param xi: x-coordinate of segment i for i=1:4
    :param yi: y-coordinate of segment i for i=1:4
    :param segment: [bool]

    :returns:
        - (x, y) tuple with x and y coordinartes of the intersection point
        - [list] error_msg list

    """
    error_msg = []
    # Default value for the intersection point
    x_intersection = 0
    y_intersection = 0
    # Check wether the x-coordinates of each segment are not the same to avoid dividing by 0
    if ((x1 == x2) or (x3 == x4)):
        if ((x1 == x2) and (x3 == x4)):
            error_msg.append('Both segments are vertical, possibly infinite intersection points, or none')
        elif (x1 == x2):
            x_intersection = x1
            a2 = (y3-y4)/(x3-x4)
            b2 = y3-a2*x3
            y_intersection = a2 * x_intersection + b2
        elif (x1 == x2):
            x_intersection = x1
            a2 = (y3-y4)/(x3-x4)
            b2 = y3-a2*x3
            y_intersection = a2 * x_intersection + b2
        else:
            raise Error('Something is wrong in this case!')

    else:
        # Find coefficients such that f = a*x+b
        a1 = (y1-y2)/(x1-x2)
        a2 = (y3-y4)/(x3-x4)
        b1 = y1-a1*x1  # = y2-a1*x2
        b2 = y3-a2*x3  # = y4-a2*x4

        # Check wether the segments are parallel:
        if (a1 == a2):
            error_msg.append('No intersection point: segments are parallel')
        else:
            # Compute intersection point: a1*x+b1 = a2*x+b2 --> x* = (b2-b1)/(a1-a2)
            x_intersection = (b2 - b1) / (a1 - a2)
            y_intersection = a1 * x_intersection + b1
            # equivalently y_intersection = a2 * x_intersection + b2
    if segment is True:
        cond_1 = x_intersection > max(min(x1, x2), min(x3, x4))
        cond_2 = x_intersection < min(max(x1, x2), max(x3, x4))
        cond_3 = y_intersection > max(min(y1, y2), min(y3, y4))
        cond_4 = y_intersection < min(max(y1, y2), max(y3, y4))
        if not (cond_1 or cond_2 or cond_3 or cond_4):
            error_msg.append('No intersection point, intersection happens out of segment bounds')
            error_msg.append('Conditions are: 1:{} 2:{} 3:{} 4:{}'.format(cond_1, cond_2, cond_3, cond_4))

    return (x_intersection, y_intersection), error_msg


# import matplotlib.pylab as plt
# import scipy.sparse as sps
def plot_sparse():
    A = sps.rand(10000, 10000, density=0.00001)
    M = sps.csr_matrix(A)
    plt.spy(M)
    plt.show()


def plot_sparsity_pattern(A, tick_frequency):
    """:param A: np array containing the matrix"""
    A[np.where(A != 0)] = 1
    plt.matshow(A, fignum=None)
    plt.colorbar()
    plt.xticks(np.arange(0, A.shape[0], tick_frequency))
    plt.yticks(np.arange(0, A.shape[0], tick_frequency))
    plt.grid()
    plt.show()


def ephemeris_bcrs(t):
    pass
