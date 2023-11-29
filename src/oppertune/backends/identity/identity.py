from typing import Iterable, Union

from ...values import CategoricalValue, ContinuousValue, DiscreteValue
from ..base import AlgorithmBackend

__all__ = ("Identity",)


class Identity(AlgorithmBackend):
    def __init__(self, parameters: Iterable[Union[CategoricalValue, ContinuousValue, DiscreteValue]], *args, **kwargs):
        self.params = tuple(parameters)

    def predict(self, *args, **kwargs):
        return {p.name: p.initial_value for p in self.params}

    def set_reward(self, *args, **kwargs):
        pass

    def __str__(self):
        return "Identity"
