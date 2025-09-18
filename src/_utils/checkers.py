from functools import wraps
from inspect import signature


def check_input_shape(func):
    @wraps(func)
    def inner(*args, **kwargs):
        bound_args = signature(func).bind(*args, **kwargs)
        bound_args.apply_defaults()
        X = bound_args.arguments.get("X")
        y = bound_args.arguments.get("y")

        if len(X.shape) < 2:
            raise ValueError("X must be a 2D array-like structure.")
        if y is None:
            return

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if len(y.shape) > 1:
            raise ValueError("y must be a 1D array-like structure.")
        
        func(*args, **kwargs)
    return inner
