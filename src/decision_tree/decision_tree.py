from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .._base.base_model import BaseModel
from .._utils.numerical import gini, mse
from .._utils.helper import partitions
from .nodes import NumericDecisionTreeNode, CategoricalDecisionTreeNode, DecisionTreeLeafNode


class DecisionTreeModel(BaseModel):
    def __init__(self):
        self._tree = None

    def fit(self, X, y, categories, numerics, stopping_criteria=None):
        self._tree = self._fit(X, y, categories, numerics, stopping_criteria)

    def _fit(self, X, y, categories, numerics, stopping_criteria=None, depth=0):
        """
        categories: list of categorical columns
        numerics: list of numerical columns
        """
        # perfectly separated
        if self._split_metric(y) == 0:
            return DecisionTreeLeafNode(self._calc_prediction(y))

        # stopping criteria
        if stopping_criteria is None:
            stopping_criteria = ("max_depth", 10)
        if stopping_criteria[0] not in ("max_depth", "min_nodes"):
            raise ValueError("Unrecognized stopping criteria")
        if stopping_criteria[0] == "max_depth" and depth >= stopping_criteria[1]:
            return DecisionTreeLeafNode(self._calc_prediction(y))
        if stopping_criteria[0] == "min_nodes" and len(y) <= stopping_criteria[1]:
            return DecisionTreeLeafNode(self._calc_prediction(y))

        # we keep track since might need to use one with second lowest gini if categorical value is missing
        # ie surrogate split.
        # Also need to think about what to do if null in training and prediction, for categorical and numerical data.
        # Also need to think about how to handle unseen categorical values when predicting
        candidates = {}
        for feature in X.columns:
            current = X[feature]
            if feature in numerics:
                current = current.sort_values().values # since we don't want indices on the next line
                midpoints = (current[:-1] + current[1:]) / 2
                for midpoint in midpoints:
                    node = NumericDecisionTreeNode(feature, midpoint)
                    score = self._calc_split_metric(node, X, y)
                    if feature not in candidates or score < candidates[feature]["score"]:
                        candidates[feature] = {"score": score, "node": node}
            elif feature in categories:
                for partition in partitions(current.unique(), 2):
                    node = CategoricalDecisionTreeNode(feature, *partition)
                    score = self._calc_split_metric(node, X, y)
                    if feature not in candidates or score < candidates[feature]["score"]:
                        candidates[feature] = {"score": score, "node": node}
            else:
                raise ValueError("column not found in categories or numerics")

        # can just keep track as we go
        split_feature = min(
            candidates,
            key=lambda item_key: candidates[item_key]["score"]
        )
        node = candidates[split_feature]["node"]
        X_left, X_right = node.split(X)

        node.left = self._fit(
            X_left,
            y[X_left.index],
            categories,
            numerics,
            stopping_criteria,
            depth + 1
        )
        node.right = self._fit(
            X_right,
            y[X_right.index],
            categories,
            numerics,
            stopping_criteria,
            depth + 1
        )
        return node

    @abstractmethod
    def _calc_prediction(self, X, y):
        pass

    @abstractmethod
    def _split_metric(self, y_left, y_right):
        """
        Must be such that a perfect split (pure node, in the case of classification
        and one whose distance metric = 0 for regression) returns zero
        """
        pass

    def _calc_split_metric(self, node, X, y):
        """weighted split metric"""
        X_left, X_right = node.split(X)
        y_left = y[X_left.index]
        y_right = y[X_right.index]
        return (
            self._split_metric(y_left) * len(y_left)
            + self._split_metric(y_right) * len(y_right)
        ) / len(y)

    def predict(self, X):
        pred = self._forward(self._tree, X)
        pred = pd.concat(pred)
        return pred[X.index]

    def _forward(self, node, X, output=None):
        """
        For prediction
        """
        if output is None:
            output = []
        if len(X) == 0: # empty branch
            return output
        if node.left is None and node.right is None:
            output.append(
                pd.Series([node.prediction] * len(X), index=X.index)
            )
            return output

        X_left, X_right = node.split(X)
        self._forward(node.left, X_left, output)
        self._forward(node.right, X_right, output)
        return output


class DecisionTreeClassifier(DecisionTreeModel):
    @staticmethod
    def _calc_prediction(y):
        return y.value_counts().sort_values().index[-1] # can optimize

    @staticmethod
    def _split_metric(y):
        return gini(y)


class DecisionTreeRegressor(DecisionTreeModel):
    @staticmethod
    def _calc_prediction(y):
        return y.mean()

    @staticmethod
    def _split_metric(y):
        return mse(y)
