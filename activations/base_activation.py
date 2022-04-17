from abc import ABC
from typing import Iterable, Union
from base_abstract import BaseSomovUnit


class BaseActivation(BaseSomovUnit, ABC):
    _local_gradient_value: Union[float, Iterable] = None

    def get_local_gradient(self) -> Union[float, Iterable]:
        return self._local_gradient_value
