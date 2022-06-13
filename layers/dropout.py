from typing import Iterable, Union

from .base_layer import BaseLayer
from casting import smart_cast

import numpy as np


class Dropout(BaseLayer):
    def __init__(self,
                 p: float = 0.6):
        super().__init__(1, 1)
        self._prob: float = p
        self._mask: np.array = None

    @smart_cast
    def forward(self, x: np.array) -> np.array:
        output_value: np.array = x

        if not self._eval:
            self._mask = np.random.randint(0, 2, size=x.shape)
            self._set_local_gradient(x)
            output_value = 1/(1 - self._prob) * self._mask * x

        return output_value

    @smart_cast
    def backward(self, dx: Union[float, Iterable]) -> Union[float, Iterable]:
        return np.multiply(self._get_local_gradient(), dx)

    def _set_local_gradient(self, x: np.array) -> np.array:
        self._local_gradient_value = 1/(1 - self._prob) * self._mask
        return self._local_gradient_value
