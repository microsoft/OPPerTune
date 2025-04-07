"""The UniformRandom algorithm."""
# ruff: noqa: ANN002, ANN003

import random
from typing import Iterable, Optional, Union

from typing_extensions import TypeVar, override

from oppertune.core.types import PredictResponse
from oppertune.core.values import Categorical, Integer, Real

from ..base import Algorithm, _PredictResponse

__all__ = ("UniformRandom",)

_ParameterValueType = TypeVar("_ParameterValueType", bound=Union[str, int, float], default=Union[str, int, float])


class UniformRandom(Algorithm[_ParameterValueType]):
    class Meta:
        supported_parameter_types = (Categorical, Integer, Real)
        requires_untransformed_parameters = False
        supports_context = True
        supports_single_reward = True
        supports_sequence_of_rewards = True

    def __init__(
        self,
        parameters: Iterable[Union[Categorical, Integer, Real]],
        *args,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(parameters, *args, **kwargs)
        self._rng = random.Random(random_seed)

    @override
    def predict(self, *args, **kwargs) -> PredictResponse[_ParameterValueType]:
        prediction, _ = self._predict(*args, **kwargs)
        return PredictResponse(prediction, request_id="")

    @override
    def _predict(self, *args, **kwargs) -> _PredictResponse[_ParameterValueType]:
        prediction = {}  # TODO Add type hint
        for param in self._raw_params:  # Using _raw_params to avoid (de)normalization overhead
            if isinstance(param, Categorical):
                val = self._rng.choice(param.categories)
            elif isinstance(param, Integer):
                val = self._rng.randint(param.min, param.max)
            else:
                assert isinstance(param, Real)
                val = self._rng.uniform(param.min, param.max)

            prediction[param.name] = val

        return _PredictResponse(prediction)

    @override
    def store_reward(self, *args, **kwargs) -> None:
        pass

    @override
    def set_reward(self, *args, **kwargs) -> None:
        self._set_reward(*args, **kwargs)

    @override
    def _set_reward(self, *args, **kwargs) -> None:
        self._iteration += 1

    @override
    def _set_reward_for_sequence(self, *args, **kwargs) -> None:
        self._iteration += 1

    @property
    @override
    def iteration(self) -> int:
        return self._iteration
