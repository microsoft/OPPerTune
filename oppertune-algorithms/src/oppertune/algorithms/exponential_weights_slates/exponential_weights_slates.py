from math import exp, inf, log, sqrt
from typing import Iterable, Optional, Tuple

import numpy as np
from typing_extensions import TypeVar, override

from oppertune.core.types import PredictResponse
from oppertune.core.values import Categorical

from ..base import Algorithm, _PredictResponse, _TuningRequest

__all__ = ("ExponentialWeightsSlates",)

_ParameterValueType = TypeVar("_ParameterValueType", bound=str, default=str)


class ExponentialWeightsSlates(Algorithm[_ParameterValueType]):
    class Meta:
        supported_parameter_types = (Categorical,)
        requires_untransformed_parameters = False
        supports_context = False
        supports_single_reward = True
        supports_sequence_of_rewards = False

    def __init__(
        self,
        parameters: Iterable[Categorical],
        random_seed: Optional[int] = None,
    ):
        super().__init__(parameters, random_seed=random_seed)
        self.params: Tuple[Categorical, ...]  # For type hints

        self.k = tuple(param.n_categories for param in self.params)
        self.eta = [sqrt(log(k) / (k * (self._iteration + 1))) for k in self.k]
        self.delta = [min(1.0, sqrt(k / (self._iteration + 1))) for k in self.k]
        self.p = [np.full([k], 1.0 / k) for k in self.k]
        self.p_hat = [np.zeros([k]) for k in self.k]

        self._pred_idx = [0 for _ in range(len(self.params))]
        self._min_reward = inf
        self._rng = np.random.default_rng(random_seed)

    @override
    def predict(self, context: None = None, predict_data: None = None) -> PredictResponse[_ParameterValueType]:
        return super().predict(context, predict_data)

    @override
    def _predict(self, context: None = None, predict_data: None = None) -> _PredictResponse[_ParameterValueType]:
        for i in range(len(self.params)):
            self.p_hat[i] = (1 - self.delta[i]) * self.p[i] + self.delta[i] / self.k[i]
            self._pred_idx[i] = int(self._rng.multinomial(n=1, pvals=self.p_hat[i]).argmax())

        prediction = {param.name: param.category(pred_idx) for param, pred_idx in zip(self.params, self._pred_idx)}
        return _PredictResponse(prediction)

    @override
    def _set_reward(self, tuning_request: _TuningRequest[_ParameterValueType]) -> None:
        reward = tuning_request.reward
        if reward is None:
            raise ValueError(f"reward cannot be None for {self.__class__.__name__}")

        if reward < self._min_reward:
            self._min_reward = reward

        for p, p_hat, pred_idx, eta in zip(self.p, self.p_hat, self._pred_idx, self.eta):
            p_sum = 1 - p[pred_idx]
            p[pred_idx] *= exp(eta * reward / p_hat[pred_idx])
            p_sum += p[pred_idx]
            p /= p_sum

        self._iteration += 1
        self._decay_eta()

    def _decay_eta(self) -> None:
        for i, k in enumerate(self.k):
            self.eta[i] = sqrt(log(k) / (k * (self._iteration + 1))) / max(1.0, abs(self._min_reward))
            self.delta[i] = min(1.0, sqrt(k / (self._iteration + 1)))

    @property
    @override
    def iteration(self) -> int:
        return self._iteration
