import numpy as np

from .base_layer_trainable import BaseLayerTrainable

from casting import smart_cast


class DenseLayer(BaseLayerTrainable):
    def __init__(self, input_units: int, output_units: int):
        super().__init__(input_size=input_units,
                         output_size=output_units)
        self.weights = np.random.normal(loc=0.0,
                                        scale=np.sqrt(2 / (input_units + output_units)),
                                        size=(input_units, output_units))
        self.bias = np.zeros(output_units)
        self.output_units = output_units
        self.input_units = input_units
        self._x = None

    @smart_cast
    def forward(self, x: np.array) -> np.array:
        self._x = x

        if not self._eval:
            # train mode
            output = np.dot(x, self.weights) + self.bias
            return output
        else:
            return np.dot(x, self.weights) + self.bias

    def _set_weight_gradients(self, dx: np.array) -> np.array:
        grad_input = np.dot(dx, self.weights.T)
        dx_ = dx.reshape(self.output_units, -1)
        grad_biases = dx.mean(axis=0) * self._x.shape[0]
        grad_weights = np.dot(dx_, self._x.reshape(-1, self.input_units))
        return grad_input, grad_weights, grad_biases.T

    def backward(self, dx: np.array) -> np.array:
        return self._set_weight_gradients(dx)
