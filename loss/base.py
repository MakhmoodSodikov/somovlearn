from abc import ABC


class BaseLoss(ABC):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError
