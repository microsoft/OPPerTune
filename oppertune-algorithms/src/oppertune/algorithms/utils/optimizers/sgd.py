import numpy as np
from typing_extensions import override

from .base import Optimizer

__all__ = ("SGD",)


class SGD(Optimizer):
    def __init__(self, eta: float):
        self.eta = eta

    @override
    def get_step_value(self, grad: np.ndarray):
        return self.eta * grad
