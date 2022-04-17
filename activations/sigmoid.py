from .base_activation import BaseActivation
import numpy as np

from casting import smart_cast


class Sigmoid(BaseActivation):
    _local_gradient_value: np.array = None

    def __init__(self):
        pass

    @smart_cast
    def forward(self, x: np.array) -> np.array:
        self._set_local_gradient(x)
        return self._sigmoid(x)

    @smart_cast
    def backward(self, x: np.array) -> np.array:
        return self.get_local_gradient()

    def _set_local_gradient(self, x: np.array) -> np.array:
        gradient_value: np.array

        gradient_value = self._sigmoid(x) * (1 - self._sigmoid(x))

        self._local_gradient_value = gradient_value

        return gradient_value

    @staticmethod
    def _sigmoid(x: np.array):
        return 1 / (1 + np.exp(x))
