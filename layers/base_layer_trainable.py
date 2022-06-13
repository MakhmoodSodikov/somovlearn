from abc import ABC, abstractmethod
from base_abstract.base_somov_unit import BaseSomovUnit
from typing import Union, Iterable


class BaseLayerTrainable(BaseSomovUnit, ABC):
    # External hyperparameters
    _eval: bool = False
    _l1_regularization: bool = False
    _l2_regularization: bool = False
    _input_size: int
    _output_size: int

    # Model trainable parameters
    weights: Union[Iterable] = None
    bias: Union[Iterable] = None

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 l1_regularization: bool = False,
                 l2_regularization: bool = False):
        self._input_size = input_size
        self._output_size = output_size
        self._l1_regularization = l1_regularization
        self._l2_regularization = l2_regularization

    @abstractmethod
    def _set_weight_gradients(self, dx: Union[Iterable]) -> Union[float, Iterable]:
        raise NotImplementedError

    def eval(self, flag: bool = True) -> bool:
        self._eval = flag
        return self._eval
