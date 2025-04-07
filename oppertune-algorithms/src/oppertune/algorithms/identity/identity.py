"""The Identity algorithm."""
# ruff: noqa: ANN002, ANN003

from typing import Iterable, Union

from typing_extensions import TypeVar, override

from oppertune.core.types import PredictResponse
from oppertune.core.values import Categorical, Integer, Real

from ..base import Algorithm, _PredictResponse

__all__ = ("Identity",)

_ParameterValueType = TypeVar("_ParameterValueType", bound=Union[str, int, float], default=Union[str, int, float])


class Identity(Algorithm[_ParameterValueType]):
    class Meta:
        supported_parameter_types = (Categorical, Integer, Real)
        requires_untransformed_parameters = False
        supports_context = True
        supports_single_reward = True
        supports_sequence_of_rewards = True

    def __init__(
        self,
        parameters: Iterable[Union[Categorical, Integer, Real]],
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ):
        super().__init__(parameters, *args, **kwargs)

    @override
    def predict(self, *args, **kwargs) -> PredictResponse[_ParameterValueType]:
        prediction, _ = self._predict(*args, **kwargs)
        return PredictResponse(prediction, request_id="")

    @override
    def _predict(self, *args, **kwargs) -> _PredictResponse[_ParameterValueType]:
        # Using _raw_params to avoid (de)normalization overhead
        prediction = {param.name: param.val for param in self._raw_params}
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
