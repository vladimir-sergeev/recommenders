from abc import ABC, abstractmethod


class Recommender(ABC):

    @abstractmethod
    def fit(self, *args):
        pass

    @abstractmethod
    def predict(self, *args):
        pass
