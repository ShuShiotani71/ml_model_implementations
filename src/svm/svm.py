from abc import abstractmethod
from functools import partial
import typing

import numpy as np
import pandas as pd
import scipy

from .._base.base_model import BaseModel
from .._utils.checkers import check_input_shape
from ._kernels import kernel_factory


class BaseSVM(BaseModel):
    def __init__(self, kernel: str | typing.Callable = "linear", c=0, **kwargs):
        # optimal, nonzero lagrange multipliers and correspinding support vectors
        self._coefs = None
        self._sv_X = None
        self._sv_y = None

        self._c = c # regularization parameter for soft margin; 0 means hard margin which is default

        self._kernel = kernel
        self._kernel_func = kernel_factory(kernel, **kwargs) if isinstance(kernel, str) else kernel

        self._default_opt_kwargs = {
                "method": "COBYQA",
                "tol": 1e-6,
            }

    @check_input_shape
    def fit(self, X, y, initial_alpha=None, **kwargs):
        def _minimization_objective_func(alpha):
            return -1 * self._objective_func(X, y, alpha)

        if initial_alpha is None:
            initial_alpha = np.zeros(len(X))
        if not kwargs:
            kwargs = self._default_opt_kwargs

        result = scipy.optimize.minimize(
            _minimization_objective_func,
            initial_alpha,
            constraints=self._constraints(y),
            bounds=self._bounds(X),
            **kwargs
        )

        # only need nonzero coefs as well as corresponding support vectors and y
        self._coefs = result.x[result.x > 0]
        self._sv_X = X[result.x > 0] # dataframe
        self._sv_y = y[result.x > 0] # series

    @check_input_shape
    def predict(self, X):
        return self._decision_func(X)

    @check_input_shape
    @abstractmethod
    def _objective_func(self, X, y, alpha):
        """
        dual formulation since we want to use kernels. hence to be maximized
        """
        pass

    @check_input_shape
    def _constraints(self, X, y) -> tuple | list:
        """
        how to do constrained optimization in scipy?
        """
        pass

    def _bounds(self, X, y) -> tuple | list:
        pass

    @check_input_shape
    @abstractmethod
    def _decision_func(self, X):
        pass


class SVMClassifier(BaseSVM):
    @check_input_shape
    def _objective_func(self, X, y, alpha):
        Y = np.outer(y, y)
        Alpha = np.outer(alpha, alpha)
        kernelized = self._kernel_func(X, X)
        # np.sum() on dataframe behaves like df.sum(), so summation is only performed over axis 0, rather than on all axes which is what happens if it were a np.array
        return np.sum(np.sum(alpha) + np.sum(Y * Alpha * kernelized))

    @check_input_shape
    def _decision_func(self, X):
        y_k_tiled = np.tile(self._sv_y.iloc[0], (len(X), 1))
        X_k_tiled = np.tile(self._sv_X.iloc[0], (len(X), 1))
        lambda_tiled = np.tile(self._coefs, (len(X), 1))
        y_tiled = np.tile(self._sv_y, (len(X), 1))

        diff_kernel = self._kernel_func(self._sv_X, X) - self._kernel_func(self._sv_X, X_k_tiled)
        # print(y_k_tiled.shape, X_k_tiled.shape, lambda_tiled.shape, y_tiled.shape, diff_kernel.shape)
        return np.sign(
            y_k_tiled.flatten()
            + np.sum(lambda_tiled.T * y_tiled.T * diff_kernel)
        )

    def _constraints(self, y):
        return (
            {"type": "eq", "fun": lambda alpha: alpha @ y}
            for i in range(len(y))
        )
    
    @check_input_shape
    def _bounds(self, X):
        upper = None if self._c == 0 else self._c
        return ((0, upper) for _ in range(len(X)))

class SVMRegressor(BaseSVM):
    def _objective_func(self, X, y):
        pass

    def _decision_func(self, X):
        pass

    def _constraints(self, X, y):
        pass
