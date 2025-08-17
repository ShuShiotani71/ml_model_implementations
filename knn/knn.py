import warnings

import numpy as np
import pandas as pd
import scipy

from .._base.base_model import BaseModel


class BasicKNN(BaseModel):
    """
    Naive implementation
    Training: O(1) time, O(1) space
    Predicting: O(nd + nk) time, O(k) space

    NB: Not saying that my implementation is of the above space-time complexity;
    I've made some assignments for better readability, sorting for convenience, etc
    which in the optimal implementation you wouldn't so just keep that in mind.
    """
    def __init__(self, k):
        self._k = k
        self._X = None
        self._y = None

    def fit(self, X, y):
        """
        could also implement add_data() which you can use to add a data point
        """
        self._check_input(X, y)
        self._X = X
        self._y = y

    def predict(self, X, dist_metric=None):
        self._check_input(X)
        if self._X is None or self._y is None:
            raise ValueError("Must fit model first.")
        self._check_input(X)
        dist_func = self._get_dist_func(dist_metric)

        res = []
        for idx, record in X.iterrows():
            distances = dist_func(record)
            min_idx = np.argpartition(distances, self._k)[:self._k]
            closest = self._y[min_idx].value_counts().sort_values()
            if closest.iloc[-1] == closest.iloc[-2]:
                warnings.warn(
                    "The majority class isn't unique; consider using a different value of k."
                )
            res.append(closest.index[-1])
        return pd.Series(res, index=X.index)

    def _get_dist_func(self, dist_metric=None):
        """
        X must be of shape(n_features,)
        Can let users pass a custom distance metric as well.
        """
        if dist_metric is None:
            dist_metric = "euclidean"
        if dist_metric == "euclidean":
            return lambda X: np.sqrt(
                ((X - self._X) ** 2).sum(axis=1)
            )
        if dist_metric == "manhattan":
            return lambda X: (X - self._X).abs().sum(axis=1)
        raise ValueError("Unrecognized dist_func.")


class OptimizedKNN(BaseModel):
    """
    Optimized implementation using KDTree
    Training: O(nlogn) time, O(n) space
    Predicting: O(klogn) time, O(k) space

    NB: Not saying that my implementation is of the above space-time complexity;
    I've made some assignments for better readability, sorting for convenience, etc
    which in the optimal implementation you wouldn't so just keep that in mind.
    """
    def __init__(self, k):
        self._k = k
        self._X = None # stored as KDTree
        self._y = None

    def fit(self, X, y, **kwargs):
        self._check_input(X, y)
        self._X = scipy.spatial.KDTree(X, **kwargs)
        self._y = y

    def predict(self, X, dist_metric=None):
        if self._X is None:
            raise ValueError("Must fit model first.")
        self._check_input(X)

        if dist_metric is None or dist_metric == "euclidean":
            p = 2
        elif dist_metric == "manhattan":
            p = 1
        else:
            raise ValueError("Unrecognized dist_metric.")

        _, idx = self._X.query(X, k=self._k, p=p) # k*log(n) time

        res = []
        for min_idx in idx:
            closest = self._y[min_idx].value_counts().sort_values()
            if closest.iloc[-1] == closest.iloc[-2]:
                warnings.warn(
                    "The majority class isn't unique; consider using a different value of k."
                )
            res.append(closest.index[-1])
        return pd.Series(res, index=X.index)
