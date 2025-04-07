"""The ExponentialWeights algorithm."""

from functools import reduce
from math import exp, inf, log, sqrt
from operator import mul
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from typing_extensions import TypeVar, override

from oppertune.core.types import PredictResponse
from oppertune.core.values import Categorical

from ..base import Algorithm, _PredictResponse, _TuningRequest

__all__ = ("ExponentialWeights",)

_ParameterValueType = TypeVar("_ParameterValueType", bound=str, default=str)


class ExponentialWeights(Algorithm[_ParameterValueType]):
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

        self.k = reduce(mul, (param.n_categories for param in self.params), 1)  # Total categorical combinations
        self.eta = sqrt(log(self.k) / (self.k * (self._iteration + 1)))
        self.delta = min(1.0, sqrt(self.k / (self._iteration + 1)))
        self.p = np.full([self.k], 1.0 / self.k)
        self.p_hat = np.zeros([self.k], dtype=float)

        self._pred_idx = -1
        self._min_reward = inf
        self._rng = np.random.default_rng(random_seed)

    @override
    def predict(self, context: None = None, predict_data: None = None) -> PredictResponse[_ParameterValueType]:
        return super().predict(context, predict_data)

    @override
    def _predict(self, context: None = None, predict_data: None = None) -> _PredictResponse[_ParameterValueType]:
        self.p_hat = (1 - self.delta) * self.p + self.delta / self.k
        self._pred_idx = int(self._rng.multinomial(n=1, pvals=self.p_hat).argmax())

        # To ensure correct ordering of parameter keys
        prediction: Dict[str, str] = {param.name: None for param in self.params}  # type: ignore

        # Get the permutation for pred_idx index
        temp_idx = self._pred_idx
        for param in reversed(self.params):
            temp_idx, param_value_idx = divmod(temp_idx, param.n_categories)
            prediction[param.name] = param.category(param_value_idx)

        return _PredictResponse(prediction)

    @override
    def _set_reward(self, tuning_request: _TuningRequest[_ParameterValueType]) -> None:
        reward = tuning_request.reward
        if reward is None:
            raise ValueError(f"reward cannot be None for {self.__class__.__name__}")

        if reward < self._min_reward:
            self._min_reward = reward

        p_sum = 1 - self.p[self._pred_idx]
        self.p[self._pred_idx] *= exp(self.eta * reward / self.p_hat[self._pred_idx])
        p_sum += self.p[self._pred_idx]
        self.p /= p_sum

        self._iteration += 1
        self._decay_eta()

    def _decay_eta(self) -> None:
        self.eta = sqrt(log(self.k) / (self.k * (self._iteration + 1))) / max(1.0, abs(self._min_reward))
        self.delta = min(1.0, sqrt(self.k / (self._iteration + 1)))

    @property
    @override
    def iteration(self) -> int:
        return self._iteration
