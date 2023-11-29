from functools import reduce
from math import exp, inf, log, sqrt
from operator import mul
from typing import Dict, Hashable, Iterable, Union

import numpy as np

from ...values import CategoricalValue
from ..base import AlgorithmBackend, PredictResponse

__all__ = ("ExponentialWeights",)


class ExponentialWeights(AlgorithmBackend):
    def __init__(self, parameters: Iterable[CategoricalValue], random_seed=None):
        self.params = tuple(parameters)
        self._round = 1
        self.min_reward = inf
        self.rng = np.random.default_rng(random_seed)

        self.k = reduce(
            mul, (param.n_categories for param in self.params), 1
        )  # Total permutations of categorical values
        self.eta = sqrt(log(self.k) / (self.k * self._round))
        self.delta = min(1.0, sqrt(self.k / self._round))
        self.p = np.full([self.k], 1.0 / self.k)
        self.p_hat = np.zeros([self.k])
        self.pred_idx = 0

    def predict(self):
        self.p_hat = (1 - self.delta) * self.p + self.delta / self.k
        self.pred_idx = int(self.rng.multinomial(n=1, pvals=self.p_hat).argmax())

        # To ensure correct ordering of parameter keys
        param_dict: Dict[str, Hashable] = {param.name: None for param in self.params}

        # Get the permutation for pred_idx index
        temp_idx = self.pred_idx
        for param in reversed(self.params):
            temp_idx, param_value_idx = divmod(temp_idx, param.n_categories)
            param_dict[param.name] = param.category(param_value_idx)

        return PredictResponse(parameters=param_dict)

    def set_reward(self, reward: Union[float, int], metadata=None):
        if reward < self.min_reward:
            self.min_reward = reward

        p_sum = 1 - self.p[self.pred_idx]
        self.p[self.pred_idx] *= exp(self.eta * reward / self.p_hat[self.pred_idx])
        p_sum += self.p[self.pred_idx]
        self.p /= p_sum

        self._round += 1
        self._eta_decay()

    def _eta_decay(self):
        self.eta = sqrt(log(self.k) / (self.k * self._round)) / max(1.0, abs(self.min_reward))
        self.delta = min(1.0, sqrt(self.k / self._round))

    @property
    def round(self):
        return self._round - 1
