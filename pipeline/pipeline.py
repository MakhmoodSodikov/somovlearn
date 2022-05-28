import numpy as np
import matplotlib.pyplot as plt

from base_abstract import BaseSomovUnit
from layers import DenseLayer
from activations import ReLU
from loss.base import BaseLoss


class Pipeline(object):
    def __init__(self):
        self._container = []
        self._optimizer = None
        self._loss: BaseLoss = BaseLoss()

    def add(self, unit: BaseSomovUnit):
        self._container.append(unit)

    def units(self):
        i = 0

        while i < len(self._container):
            yield self._container[i]
            i += 1

    def reverse(self):
        self._container.reverse()

    def compile(self, optimizer, loss):
        self._optimizer = optimizer
        self._loss = loss

    def _visualize(self,
                   train_step_history,
                   test_step_history):
        pass

    def _train_step(self,
                    features_train,
                    targets_train):
        history = {}

        # do smth

        return history

    def _validation_step(self,
                         features_test,
                         targets_test):
        history = {}

        # do smth

        return history

    def train(self,
              features_train: np.array,
              targets_train: np.array,
              features_test: np.array,
              targets_test: np.array,
              epochs: int,
              visualize: bool = True):
        for epoch in range(epochs):
            train_step_history = self._train_step(features_train,
                                                  targets_train)
            test_step_history = self._validation_step(features_test,
                                                      targets_test)

            if visualize:
                self._visualize(train_step_history,
                                test_step_history)


if __name__ == '__main__':
    pip = Pipeline()
    pip.add(DenseLayer(10, 20))
    pip.add(ReLU())

    pip.reverse()
    pip.reverse()

    for ls in pip.units():
        print(ls)

    pip.reverse()
    print()

    for ls in pip.units():
        print(ls)
