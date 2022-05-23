from typing import Iterable, Union

from .base_layer_trainable import BaseLayerTrainable
from casting import smart_cast

import numpy as np


class BatchNorm(BaseLayerTrainable):
    def __init__(self,
                 l1_regularization: bool = False,
                 l2_regularization: bool = False):
        output_size = 0
        input_size = 0
        super().__init__(input_size, output_size, l1_regularization, l2_regularization)
        self.weights = np.array([1, 0])
        self.running_mean = 1
        self.running_covariance = 0
        self._first_run = True
        self._covariance: np.array = np.array([])
        self._mean: np.array = np.array([])
        self._normalized_x: np.array = np.array([])
        self._eps: float = 0.0000001
        self._xmu: np.array = np.array([])
        self._sqrt_var: np.array = np.array([])
        self._ivar: np.array = np.array([])


    @smart_cast
    def forward(self, x: np.array) -> np.array:
        self._input_size = x.shape[:-1]

        if not self._eval:
            mean = x.mean(axis=0)
            self._covariance = np.square(x - mean).mean(axis=0)
            self._xmu = x - mean
            self._normalized_x = (x - mean)/np.sqrt(self._covariance + self._eps)
            output_value = self._normalized_x * self.weights[0] + self.weights[1]

            if self._first_run:
                self.running_mean = mean
                self.running_covariance = self._covariance

            self.running_mean = (mean.mean() + self.running_mean) / 2
            self.running_covariance = (self._covariance.mean() + self.running_covariance) / 2

            self._sqrt_var = np.sqrt(self._covariance + self._eps)
            self._ivar = 1./self._sqrt_var
        else:
            normalized_x = (x - self.running_mean) / np.sqrt(self.running_covariance + self._eps)
            output_value = normalized_x * self.weights[0] + self.weights[1]

        return output_value

    @smart_cast
    def backward(self, dx: Union[float, Iterable]) -> Union[float, Iterable]:
        self._set_local_gradient(dx)
        return self._set_local_gradient(dx)

    def _set_local_gradient(self, dx: np.array) -> np.array:
        # get the dimensions of the input/output
        N = dx.shape[-1]
        # step9
        dbeta = np.sum(dx, axis=0)
        dgammax = dx  # not necessary, but more understandable

        # step8
        dgamma = np.sum(dgammax * self._normalized_x, axis=0)
        dxhat = dgammax * self.weights[0]

        # step7
        divar = np.sum(dxhat * self._xmu, axis=0)
        dxmu1 = dxhat * self._ivar

        # step6
        dsqrtvar = -1. / (self._sqrt_var ** 2) * divar

        # step5
        dvar = 0.5 * 1. / np.sqrt(self._covariance + self._eps) * dsqrtvar

        # step4
        dsq = 1. / N * dvar

        # step3
        dxmu2 = 2 * self._xmu * dsq

        # step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)

        # step1
        dx2 = 1. / N * dmu

        # step0
        dx = dx1 + dx2

        return dx, dgamma.mean(), dbeta.mean()
