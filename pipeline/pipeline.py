import numpy as np
import matplotlib.pyplot as plt
from typing import List

from base_abstract import BaseSomovUnit
from layers import BaseLayerTrainable, BaseLayer
from layers import DenseLayer, BatchNorm, Dropout
from activations import ReLU, Sigmoid, BaseActivation, Tanh
from loss import BaseLoss
from loss import MarginLoss, HingeLoss, MSELoss
from optim import SGDOptimizer, BaseOptim
from data.dataloader import DataLoader


class Pipeline(object):
    def __init__(self):
        self._container: List[BaseSomovUnit] = []
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
        self._optimizer: BaseOptim = optimizer
        self._loss = loss

    def _visualize(self,
                   train_step_history,
                   test_step_history):
        pass

    def backward(self, target_pred, target_real):
        loss = self._loss(target_pred, target_real)
        self.reverse()
        out_hidden: np.array = self._loss.backward()

        for unit in self._container:
            if isinstance(unit, BaseActivation) or isinstance(unit, BaseLayer):
                out_hidden = unit.backward(out_hidden)

            if isinstance(unit, BaseLayerTrainable):
                unit.eval(False)
                if isinstance(unit, BatchNorm):
                    dx_out, dgamma, dbeta = unit.backward(out_hidden)
                    unit.weights[0] = self._optimizer.step(np.array(unit.weights[0]), dgamma)
                    unit.weights[1] = self._optimizer.step(np.array(unit.weights[1]), dbeta)
                    out_hidden = dx_out
                if isinstance(unit, DenseLayer):
                    dx_out, dweights, dbias = unit.backward(out_hidden)
                    unit.weights = self._optimizer.step(unit.weights, dweights)
                    unit.bias = self._optimizer.step(unit.bias, dbias)
                    out_hidden = dx_out

        self.reverse()

        return out_hidden, loss

    def forward(self, features: np.array):
        out_hidden: np.array = features

        for unit in self._container:
            out_hidden = unit.forward(out_hidden)

        return out_hidden

    def _train_step(self,
                    features,
                    targets):
        history = {}

        _pred = self.forward(features)

        return history

    def _validation_step(self,
                         features,
                         targets):
        history = {}

        # do smth

        return history

    def predict(self,
                features):
        for batch in features:
            self.forward(batch)

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
    data, _, labels_hot = DataLoader().load_cifar10()
    pip = Pipeline()
    pip.add(DenseLayer(3072, 1024))
    pip.add(ReLU())
    pip.add(BatchNorm())
    pip.add(DenseLayer(1024, 256))
    pip.add(Tanh())
    pip.add(DenseLayer(256, 10))
    pip.add(Tanh())
    pip.compile(loss=HingeLoss(), optimizer=SGDOptimizer(learning_rate=0.0001))

    for epoch in range(25):
        #print(data[:3].shape, labels_hot[:3].shape)
        y_pred = pip.forward(data[:3])
        #print(labels_hot[:7].shape)
        loss = HingeLoss()
        print(loss(y_pred, labels_hot[:3]).mean())
        #pip.backward(y_pred[:4], labels_hot[:4])
        pip.backward(y_pred, labels_hot[:3])
        print('EPOCH_{}'.format(epoch))
