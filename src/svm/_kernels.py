from functools import partial

import numpy as np


# could use factory method pattern
# but here the only thing we gonna do with these
# kernel functions is just call them which means
# that that is the common interface and there's no
# other behavior we want to add so what we have seems fine
# more importantly here, its not like were doing anything with
# these kernels, ie calling their methods inside the factory
# alternatively, we can just make the user pass their desired
# kernel directly to the svm.
# could also use the registry pattern to make it easier to add kernels
def kernel_factory(name, **kwargs):
    """
    All kernel func must compute the kernel for every combination
    of vectors, ie a matrix product.
    """
    kernel_mapping = {
        "linear": _linear_kernel,
        "polynomial": _polynomial_kernel,
        "rbf": _rbf_kernel,
        "tanh": _tanh_kernel,
    }
    return partial(kernel_mapping[name], **kwargs)


def _linear_kernel(X1, X2):
    return X1 @ X2.T


def _polynomial_kernel(X1, X2, offset=False, d=2):
    return (X1 @ X2.T + int(offset)) ** d


def _rbf_kernel(X1, X2, sigma=1):
    Diff = X1 - X2
    return np.exp(-0.5 * (Diff @ Diff.T) / sigma)


def _tanh_kernel(X1, X2, alpha=1, c=1):
    return np.tanh(alpha * X1 @ X2.T + c)
