from abc import abstractmethod

import numpy as np
import pandas as pd
import scipy

from .._base.base_model import BaseModel


class LinearModel(BaseModel):
    def __init__(self):
        self._coefs = None

    def predict(self, X):
        X = pd.concat(
            [pd.Series(np.ones(len(X)), index=X.index), X],
            axis=1
        )
        return X @ self._coefs


class LinearRegression(LinearModel):
    def fit(self, X, y):
        X = pd.concat(
            [pd.Series(np.ones(len(X)), index=X.index), X],
            axis=1
        )
        coefs = np.linalg.inv(X.T @ X) @ X.T @ y
        self._coefs = coefs.values.reshape(-1, 1)


class GeneralizedLinearModel(LinearModel):
    def __init__(self):
        self._default_opt_kwargs = {
                "method": "L-BFGS-B",
                "tol": 1e-6,
            }
        super().__init__()

    def fit(self, X, y, initial_coefs=None, **kwargs):
        if initial_coefs is not None and X.shape[1] != len(initial_coefs) - 1:
            raise ValueError("The number of features of X must be len(coefs)-1.")

        X = pd.concat(
            [pd.Series(np.ones(len(X)), index=X.index), X],
            axis=1
        )
        if initial_coefs is None:
            # apparently R sets default coef values to zero and is often a good choice
            initial_coefs = np.zeros(X.shape[1])

        def _minimization_objective_func(initial_coefs):
            Xb = X @ initial_coefs.reshape(-1, 1)
            return -1 * self._objective_func(Xb, y)
        
        if not kwargs:
            kwargs = self._default_opt_kwargs
        result = scipy.optimize.minimize(
            _minimization_objective_func,
            initial_coefs,
            **kwargs
        )
        self._coefs = result.x.reshape(-1, 1)

    def predict(self, X):
        return self._pred_func(super().predict(X))

    @abstractmethod
    def _objective_func(self, Xb, y):
        """
        Could optimize in different ways, eg MLE.
        If MLE then its the log-likelihood function.
        """
        pass

    @abstractmethod
    def _pred_func(self, Xb):
        pass


class LogisticRegression(GeneralizedLinearModel):
    @staticmethod
    def _objective_func(Xb, y):
        """
        Simplified log likelihood function for Bernoulli distribution
        """
        return (y @ Xb - np.log(1 + np.exp(Xb))).sum()

    @staticmethod
    def _pred_func(Xb):
        return np.exp(Xb) / (1 + np.exp(Xb))


class PoissonRegression(GeneralizedLinearModel):
    def _objective_func(self, Xb, y):
        """
        Simplified log likelihood function for Poisson distribution
        """
        return (y @ Xb - np.exp(Xb)).sum()

    @staticmethod
    def _pred_func(Xb):
        return np.exp(Xb)
