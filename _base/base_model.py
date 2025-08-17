from abc import ABC, abstractmethod


# TODO: type hints
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        self._check_input(X, y)

    @abstractmethod
    def predict(self, X):
        self._check_input(X)

    def _check_input(self, X, y=None):
        if len(X.shape) < 2:
            raise ValueError("X must be a 2D array-like structure.")
        if y is None:
            return

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if len(y.shape) > 1:
            raise ValueError("y must be a 1D array-like structure.")