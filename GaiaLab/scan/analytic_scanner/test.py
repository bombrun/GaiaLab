import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad


def taylor_sine(x):  # Taylor approximation to sine function
    ans = currterm = x
    i = 0
    while np.abs(currterm) > 0.001:
        currterm = -currterm * x**2 / ((2 * i + 3) * (2 * i + 2))
        ans = ans + currterm
        i += 1
    return ans


grad_sine = grad(taylor_sine)
# print("Gradient of sin(pi) is {}".format(grad_sine(np.pi)))


def double_sin(x, y):
    return np.sin(x) + np.cos(y)


my_grad = grad(double_sin)
print(my_grad(np.pi, 2*np.pi))
