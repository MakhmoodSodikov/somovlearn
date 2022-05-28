import numpy as np
from base import BaseLoss


class MarginLoss(BaseLoss):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        A, B = args[0], args[1]
        return np.max(np.multiply(A, B)).mean()


if __name__ == '__main__':
    print(max([0, 0, 0],
                  1 + np.multiply([1, -1, 1],
                                  [0.9, 0.6, 0.1])))
