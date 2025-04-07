from .base import Optimizer
from .rmsprop import RMSprop
from .sgd import SGD

_OPTIMIZERS = {
    "rmsprop": RMSprop,
    "sgd": SGD,
}


def create_optimizer(name: str, **kwargs) -> Optimizer:
    return _OPTIMIZERS[name](**kwargs)
