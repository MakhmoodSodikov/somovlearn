import numpy as np

from .base_layer_trainable import BaseLayerTrainable
from typing import Tuple

from casting import smart_cast


class DenseLayer(BaseLayerTrainable):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size=input_size,
                         output_size=output_size)
        # self.weights = np.random.random(size=(output_size, input_size))
        self.weights = np.random.normal(loc=0.0,
                                        scale=np.sqrt(2 / (input_size + output_size)),
                                        size=(input_size, output_size))
        self.bias = np.zeros(output_size)
        self.dweights = []
        self.dbias = []


    @smart_cast
    def forward(self, x: np.array) -> np.array:
        if not self._eval:
            # train mode
            self._set_local_gradient(x)
            return np.dot(x, self.weights) + self.bias
        else:
            return np.dot(x, self.weights) + self.bias

    def _set_local_gradient(self, x: np.array) -> np.array:
        self.local_gradient_value = 0

    def backward(self, dx: np.array) -> np.array:
        pass

