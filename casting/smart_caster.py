from typing import Any, Callable, ClassVar
import numpy as np


class SmartCaster(object):
    def __init__(self, origin: ClassVar):
        self._origin = origin

    def to_origin(self, x: Any):
        return self._origin(x)


def smart_cast(method: Callable):
    _numpy_array_caster = SmartCaster(np.array)

    def smart_cast_wrapper(self, x: np.array):
        x = _numpy_array_caster.to_origin(x)
        return method(self, x)
    return smart_cast_wrapper
