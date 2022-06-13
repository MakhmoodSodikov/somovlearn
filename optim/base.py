from abc import ABC, abstractmethod


class BaseOptim(ABC):
    @abstractmethod
    def step(self, weight, dweight):
        raise NotImplementedError
