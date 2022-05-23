from abc import ABC, abstractmethod


class BaseOptim(ABC):
    @abstractmethod
    def step(self):
        raise NotImplementedError
