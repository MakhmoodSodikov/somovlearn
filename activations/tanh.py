from .base_activation import BaseActivation
import numpy as np


from casting import smart_cast


class Tanh(BaseActivation):
    _local_gradient_value: np.array = None

    def __init__(self):
        pass

    @smart_cast
    def forward(self, x: np.array) -> np.array:
        self._set_local_gradient(x)
        return np.tanh(x)

    @smart_cast
    def backward(self, dx: np.array) -> np.array:
        return np.multiply(dx, self.get_local_gradient())

    def _set_local_gradient(self, x: np.array) -> np.array:
        gradient_value: np.array

        gradient_value = 1 - np.square(np.tanh(x))

        self._local_gradient_value = gradient_value

        return gradient_value
