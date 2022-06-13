from .base import BaseOptim
import numpy as np

class SGDOptimizer(BaseOptim):
    def __init__(self, learning_rate):
        self._lr = learning_rate

    def step(self, weight, dweight):
        # TODO: Problem is here
        #dweight = dweight.T

        new_weight = weight - self._lr * dweight.T
        return new_weight
