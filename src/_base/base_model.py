from abc import ABC, abstractmethod


# TODO: type hints
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        self._check_input(X, y)

    @abstractmethod
    def predict(self, X):
        self._check_input(X)
