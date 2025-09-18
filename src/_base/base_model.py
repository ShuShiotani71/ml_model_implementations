from abc import ABC, abstractmethod


# TODO: type hints
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
