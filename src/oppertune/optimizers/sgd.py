import numpy as np

from .base import Optimizer

__all__ = ("SGD",)


class SGD(Optimizer):
    def __init__(self, eta: float):
        self.eta = eta

    def get_step_value(self, grad: np.ndarray):
        return self.eta * grad
