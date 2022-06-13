import numpy as np
from .base import BaseLoss


class MarginLoss(BaseLoss):
    def __init__(self, margin: float = 1.):
        self._margin: float = margin

    def __call__(self, *args, **kwargs):
        y_pred, y = np.array(args[0]), np.array(args[1])
        num_classes = np.max(y) + 1
        li = 0
        loss = 0

        for i, y_batch in enumerate(y):
            for k, y_true in enumerate(y_batch):
                for j in range(num_classes):
                    true_class = y_true[j]
                    if true_class != 1:
                        pred_class = y_pred[i][k][j]
                        li += np.maximum([0.], pred_class - true_class + self._margin)
                loss += li

        return loss/(y.shape[0] * y.shape[1])


if __name__ == '__main__':
    y = [
            [
                [-1, 1],
                [-1, 1]
            ],
            [
                [-1, 1],
                [1, -1]
            ]
        ]

    y_pred = [
        [
            [1, 1],
            [-1, 1]
        ],
        [
            [-1, 1],
            [1, -1]
        ]
        ]

    loss = MarginLoss()
    print(loss(y_pred, y))
