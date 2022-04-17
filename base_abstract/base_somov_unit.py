from typing import Iterable, Union
from abc import ABC, abstractmethod


class BaseSomovUnit(ABC):
    @abstractmethod
    def forward(self, x: Union[Iterable]) -> Union[Iterable]:
        raise NotImplementedError

    @abstractmethod
    def backward(self, dx: Union[Iterable]) -> Union[Iterable]:
        raise NotImplementedError
