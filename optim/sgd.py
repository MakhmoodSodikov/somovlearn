from .base import BaseOptim


class SGDOptimizer(BaseOptim):
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def step(self):
        pass
