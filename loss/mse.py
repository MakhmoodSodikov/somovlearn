import numpy as np
from .base import BaseLoss


class MSELoss(BaseLoss):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        A, B = args[0], args[1]
        return np.square(np.subtract(A, B)).mean()
