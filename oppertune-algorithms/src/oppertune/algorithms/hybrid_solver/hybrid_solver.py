"""The Hybrid Solver algorithm."""

from typing import Any, Dict, Iterable, Optional, Union

from typing_extensions import TypeVar, override

from oppertune.core.types import PredictResponse
from oppertune.core.values import Categorical, Integer, Real

from ..base import Algorithm, _PredictResponse, _TuningRequest

__all__ = ("HybridSolver",)

_ParameterValueType = TypeVar("_ParameterValueType", bound=Union[str, int, float], default=Union[str, int, float])


class HybridSolver(Algorithm[_ParameterValueType]):
    class Meta:
        supported_parameter_types = (Categorical, Integer, Real)
        requires_untransformed_parameters = False
        supports_context = False
        supports_single_reward = True
        supports_sequence_of_rewards = False

    def __init__(
        self,
        parameters: Iterable[Union[Categorical, Integer, Real]],
        categorical_algorithm: str = "exponential_weights",
        numerical_algorithm: str = "bluefin",
        categorical_algorithm_args: Optional[Dict[str, Any]] = None,
        numerical_algorithm_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(parameters)

        self.categorical_params = tuple(param for param in self.params if isinstance(param, Categorical))
        self.numerical_params = tuple(param for param in self.params if isinstance(param, (Integer, Real)))

        from ..all import create_tuning_instance  # To avoid circular import

        self.categorical_tuning_instance = (
            create_tuning_instance(self.categorical_params, categorical_algorithm, categorical_algorithm_args)
            if self.categorical_params
            else None
        )

        self.numerical_tuning_instance = (
            create_tuning_instance(self.numerical_params, numerical_algorithm, numerical_algorithm_args)
            if self.numerical_params
            else None
        )

    @override
    def predict(self, context: None = None, predict_data: None = None) -> PredictResponse[_ParameterValueType]:
        return super().predict(context, predict_data)

    @override
    def _predict(self, context: None = None, predict_data: None = None) -> _PredictResponse[_ParameterValueType]:
        _prediction = {}

        if self.categorical_tuning_instance is not None:
            categorical_prediction, _ = self.categorical_tuning_instance.predict()
            _prediction.update(categorical_prediction)

        if self.numerical_tuning_instance is not None:
            numerical_prediction, _ = self.numerical_tuning_instance.predict()
            _prediction.update(numerical_prediction)

        prediction = {param.name: _prediction[param.name] for param in self.params}
        return _PredictResponse(prediction)

    @override
    def _set_reward(self, tuning_request: _TuningRequest[_ParameterValueType]) -> None:
        reward = tuning_request.reward
        if reward is None:
            raise ValueError(f"reward cannot be None for {self.__class__.__name__}")

        context_id = tuning_request.context.id

        if self.categorical_tuning_instance:
            self.categorical_tuning_instance.set_reward(reward, context_id)

        if self.numerical_tuning_instance:
            self.numerical_tuning_instance.set_reward(reward, context_id)

        self._iteration += 1

    @property
    @override
    def iteration(self) -> int:
        return self._iteration
