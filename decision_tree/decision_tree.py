from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .._base.base_model import BaseModel
from .._utils.numerical import gini
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
        if stopping_criteria is None:
            stopping_criteria = ("max_depth", 10)
        if stopping_criteria not in ("max_depth", "min_nodes"):
            raise ValueError("Unrecognized stopping criteria")
        if stopping_criteria[0] == "max_depth" and depth >= stopping_criteria[1]:
            return DecisionTreeLeafNode(self._calc_prediction(y))
        if stopping_criteria[0] == "min_nodes" and len(y) <= stopping_criteria[1]:
            return DecisionTreeLeafNode(self._calc_prediction(y))

        # we keep track since might need to use one with second lowest gini if categorical value us missing
        gini = {}
        for feature in X.columns:
            if feature in numerics:
                current = X[feature].sort_values()
                midpoints = (current[:-1] + current[1:]) / 2
                for midpoint in midpoints:
                    node = NumericDecisionTreeNode(feature, midpoint)
                    gini_ = self._calc_gini(node, X, y)
                    if feature not in gini or gini_ < gini[feature]["coef"]:
                        gini[feature] = {"coef": gini_, "node": node}

            elif feature in categories:
                pass # TODO: implement this

            else:
                raise ValueError("column not found in categories or numerics")

        split_feature = min(gini, key=lambda item_key: gini[item_key]["coef"])
        node = gini[split_feature]["node"]

        X_left, X_right = node.split(X)
        node.left = self._fit(X_left, y[X_left.index], categories, numerics, stopping_criteria, depth + 1)
        node.right = self._fit(X_right, y[X_right.index], categories, numerics, stopping_criteria, depth + 1)
        return node

    @abstractmethod
    def _calc_prediction(self, X, y):
        pass

    # might move this to nodes.py as a function as it depends on the node class
    @staticmethod
    def _calc_gini(node, X, y):
        """weighted gini coefficient"""
        X_left, X_right = node.split(X)
        y_left = y[X_left.index]
        y_right = y[X_right.index]

        return (
            gini(y_left) * len(y_left) + gini(y_right) * len(y_right)
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


class DecisionTreeRegressor(DecisionTreeModel):
    @staticmethod
    def _calc_prediction(y):
        return y.mean()
