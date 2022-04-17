from abc import ABC, abstractmethod
from base_abstract.base_somov_unit import BaseSomovUnit
from typing import Union, Iterable, Optional


class BaseLayer(BaseSomovUnit, ABC):
    # Internal variables
    _local_gradient_value: Union[Iterable] = None

    # External hyperparameters
    _eval: bool = False
    _input_size: int = 0
    _output_size: int = 0

    def __init__(self,
                 input_size: int,
                 output_size: int):
        self._input_size = input_size
        self._output_size = output_size

    @abstractmethod
    def _set_local_gradient(self, x: Union[Iterable]) -> Union[Iterable]:
        raise NotImplementedError

    def _get_local_gradient(self) -> Union[Iterable]:
        return self._local_gradient_value

    def eval(self, flag: bool = True) -> bool:
        self._eval = flag
        return self._eval
