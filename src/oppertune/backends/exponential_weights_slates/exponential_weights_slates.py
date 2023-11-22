from math import exp, inf, log, sqrt
from typing import Iterable, Union

import numpy as np

from ...values import CategoricalValue
from ..base import AlgorithmBackend, PredictResponse

__all__ = ("ExponentialWeightsSlates",)


class ExponentialWeightsSlates(AlgorithmBackend):
    def __init__(self, parameters: Iterable[CategoricalValue], random_seed=None):
        self.params = tuple(parameters)
        self._round = 1
        self.min_reward = inf
        self.rng = np.random.default_rng(random_seed)

        self.k = tuple(param.n_categories for param in self.params)
        self.eta = [sqrt(log(k) / (k * self._round)) for k in self.k]
        self.delta = [min(1.0, sqrt(k / self._round)) for k in self.k]
        self.p = [np.full([k], 1.0 / k) for k in self.k]
        self.p_hat = [np.zeros([k]) for k in self.k]
        self.pred_idx = [0 for _ in range(len(self.params))]

    def predict(self):
        for i in range(len(self.params)):
            self.p_hat[i] = (1 - self.delta[i]) * self.p[i] + self.delta[i] / self.k[i]
            self.pred_idx[i] = int(self.rng.multinomial(n=1, pvals=self.p_hat[i]).argmax())

        parameters = {param.name: param.category(pred_idx) for param, pred_idx in zip(self.params, self.pred_idx)}
        return PredictResponse(parameters=parameters)

    def set_reward(self, reward: Union[float, int], metadata=None):
        if reward < self.min_reward:
            self.min_reward = reward

        for p, p_hat, pred_idx, eta in zip(self.p, self.p_hat, self.pred_idx, self.eta):
            p_sum = 1 - p[pred_idx]
            p[pred_idx] *= exp(eta * reward / p_hat[pred_idx])
            p_sum += p[pred_idx]
            p /= p_sum

        self._round += 1
        self._eta_decay()

    def _eta_decay(self):
        for i, k in enumerate(self.k):
            self.eta[i] = sqrt(log(k) / (k * self._round)) / max(1.0, abs(self.min_reward))
            self.delta[i] = min(1.0, sqrt(k / self._round))

    @property
    def round(self):
        return self._round - 1
