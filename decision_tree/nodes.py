from abc import ABC, abstractmethod


class DecisionTreeLeafNode:
    """
    Make leaf nodes a separate class from intermediate nodes for better interface segregation.
    """
    def __init__(self, prediction):
        self.prediction = prediction # majority label if classification, mean if regression
        self.left = None
        self.right = None


class DecisionTreeInternalNode(ABC):
    def __init__(self, feature, left, right):
        self._feature = feature
        self.left = left
        self.right = right

    @abstractmethod
    def split(self, X):
        pass


class NumericDecisionTreeNode(DecisionTreeInternalNode):
    def __init__(self, feature, threshold, left=None, right=None):
        self._threshold = threshold
        super().__init__(feature, left, right)

    def split(self, X):
        # Might be better to create copy
        X_right = X[X[self._feature] > self._threshold]
        X_left = X[X[self._feature] <= self._threshold]
        return X_left, X_right


class CategoricalDecisionTreeNode(DecisionTreeInternalNode):
    def __init__(self, feature, left_values, right_values, left=None, right=None):
        self._left_values = left_values
        self._right_values = right_values
        super().__init__(feature, left, right)

    def split(self, X):
        # Might be better to create copy
        X_right = X[X[self._feature].isin(self._right_values)]
        X_left = X[X[self._feature].isin(self._left_values)]
        return X_left, X_right
