from abc import ABC


class BaseLoss(ABC):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def forward(self, target_pred, target_true):
        raise NotImplementedError
