# # test.py
#
# test for the autograd module
#
# #
from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad
from autograd import jacobian

from autograd.extend import primitive, defvjp
from autograd.test_util import check_grads


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


def double_sin(x):
    out = np.array([np.sin(x[0]) + np.cos(x[1]), 1])
    return out  # np.append(out, 2)


# print(double_sin( np.array([1,2]) ))
my_grad = jacobian(double_sin)
print(my_grad(np.array([np.pi, 2*np.pi])))


################################################################################
@primitive
def logsumexp(x):
    """Numerically stable log(sum(exp(x))), also defined in scipy.misc"""
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

# Next, we write a function that specifies the gradient with a closure.
# The reason for the closure is so that the gradient can depend
# on both the input to the original function (x), and the output of the
# original function (ans).


def logsumexp_vjp(ans, x):
    # If you want to be able to take higher-order derivatives, then all the
    # code inside this function must be itself differentiable by Autograd.
    # This closure multiplies g with the Jacobian of logsumexp (d_ans/d_x).
    # Because Autograd uses reverse-mode differentiation, g contains
    # the gradient of the objective w.r.t. ans, the output of logsumexp.
    # This returned VJP function doesn't close over `x`, so Python can
    # garbage-collect `x` if there are no references to it elsewhere.
    x_shape = x.shape
    return lambda g: np.full(x_shape, g) * np.exp(x - np.full(x_shape, ans))


# Now we tell Autograd that logsumexmp has a gradient-making function.
defvjp(logsumexp, logsumexp_vjp)

if __name__ == '__main__':
    # Now we can use logsumexp() inside a larger function that we want
    # to differentiate.
    def example_func(y):
        z = y**2
        lse = logsumexp(z)
        return np.sum(lse)

    grad_of_example = grad(example_func)
    print("Gradient: \n", grad_of_example(npr.randn(10)))

    # Check the gradients numerically, just to be safe.
check_grads(example_func, modes=['rev'])(npr.randn(10))
