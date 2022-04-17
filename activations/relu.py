import numpy as np

from .base_activation import BaseActivation

from casting import smart_cast


class ReLU(BaseActivation):
    _local_gradient_value: np.array = None

    def __init__(self):
        pass

    @smart_cast
    def forward(self, x: np.array) -> np.array:
        self._set_local_gradient(x)
        return (np.abs(x) + x) / 2

    @smart_cast
    def backward(self, dout: np.array) -> np.array:
        return self.get_local_gradient()

    def _set_local_gradient(self, x: np.array) -> np.array:
        gradient_value = x.copy()
        gradient_value[gradient_value <= 0] = 0
        gradient_value[gradient_value > 0] = 1

        self._local_gradient_value = gradient_value

        return gradient_value
