# # File for the global solutions
#
# Luca Zampieri 2018
# #
import numpy as np

num_sources = 3
num_parameters_per_sources = 5  # the astronomic parameters
s_vector = np.zeros((num_sources*num_parameters_per_sources, 1))


N_ss = np.zeros((s_vector*np.transpose(s_vector)).shape)
print('The shape of N_ss is {}'.format(N_ss.shape))
for i in range(N_ss.shape[0]):
    for j in range(N_ss.shape[1]):
        # for loop can be removed by looping just over i and saying N_ss[i,i]
        # (i.e. no need for "if" neither)
        if i == j:
            N_ss[i, j] = 0  #
        else:
            N_ss[i, j] = 0
# N_aa =


def eta_obs():
    return 3


def f_obs():  # xi_obs():
    return 3


def f_calc():
    return 3


def eta0_fng(mu, f, n, g):
    # = eta0_ng
    # TODO: define Y_FPA, F
    return -Y_FPA[n, g]/F


def xi0_fng(mu, f, n, g):
    """
    :attribute X_FPA[n]: physical AC coordinate of the nominal center of the nth CCD
    :attribute Y_FPA[n,g]: physical AL coordinate of the nominal observation line for gate g on the nth CCD
    :attribute Xcentre_FPA[f]:
    """
    mu_c = 996.5
    p_AC = 30  # [micrometers]
    return -(X_FPA[n] - (mu - mu_c) * p_AC - Xcenter_FPA[f])/F


def R_l(s, a, c, g):
    return f_obs() - f_calc()


class AGIS:

    def __init__(self):
        self.a = 0

    def update_S_block(self):
        for i in num_sources:
            self.update_source(i)

    def update_source(self, i):
        # compute A_i
        # compute h_i
        # compute d_i
        d_i = np.zeros((5, 1))
        return d_i
