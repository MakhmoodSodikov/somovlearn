import numpy as np
from .base import BaseLoss


class HingeLoss(BaseLoss):
    '''
    L1 Hinge Loss implementation
    Reference: http://juliaml.github.io/LossFunctions.jl/stable/losses/margin/
    '''

    def __init__(self):
        self._local_gradient: np.array = None
        self._base = [0]

    def forward(self, target_pred, target_true):
        agreement = target_pred * target_true
        loss = np.maximum(self._base, 1 - agreement)
        dloss = np.zeros_like(agreement)
        dloss[agreement < 1] = -1
        # print(dloss.shape)
        self._local_gradient = dloss
        return loss

    def backward(self):
        return self._local_gradient

    def __call__(self, *args, **kwargs):
        target_pred, target_true = np.array(args[0]), np.array(args[1])
        return self.forward(target_pred, target_true)


if __name__ == "__main__":
    loss = HingeLoss()
    y = np.array([
        [
            [-1, 1],
            [-1, 1]
        ],
        [
            [-1, 1],
            [1, -1]
        ]
    ])

    y_pred = np.array([
        [
            [1, 1],
            [-1, 1]
        ],
        [
            [-1, 1],
            [1, -1]
        ]
    ])

    loss.forward(y_pred, y)

