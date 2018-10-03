# # Helper functions for the analytic scanner
#
# Contains:
#   - compute_intersection
#   - compute_angle
# LucaZampieri 2018

import numpy as np


def compute_angle(v1, v2):
    """
    Computes the angle between two Vectors
    :param vi: vector between which you want to compute the angle for each i=1:2
    :returns: [float] [deg] angle between the vectors
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


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