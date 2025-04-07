"""The Bayesian Optimization algorithm."""

from contextlib import contextmanager
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
from skopt import Optimizer, space
from typing_extensions import TypeVar, override

from oppertune.core.types import PredictResponse
from oppertune.core.values import Categorical, Integer, Real

from ..base import Algorithm, _PredictResponse, _TuningRequest

__all__ = ("BayesianOptimization",)

_ParameterValueType = TypeVar("_ParameterValueType", bound=Union[str, int, float], default=Union[str, int, float])


class BayesianOptimization(Algorithm[_ParameterValueType]):
    class Meta:
        supported_parameter_types = (Categorical, Integer, Real)
        requires_untransformed_parameters = False
        supports_context = False
        supports_single_reward = True
        supports_sequence_of_rewards = False

    def __init__(
        self,
        parameters: Iterable[Union[Categorical, Integer, Real]],
        optimizer_args: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
    ):
        super().__init__(parameters, random_seed=random_seed)
        self.params: Tuple[Union[Categorical, Integer, Real], ...]  # For type hints

        if optimizer_args is None:
            optimizer_args = {}

        if "dimensions" in optimizer_args:
            raise ValueError("Use parameters to specify bounds")

        with _substitute_numpy_int():
            dimensions = tuple(_to_skopt(param) for param in self.params)

        self.optimizer = Optimizer(dimensions, random_state=random_seed, **optimizer_args)

    @override
    def predict(self, context: None = None, predict_data: None = None) -> PredictResponse[_ParameterValueType]:
        return super().predict(context, predict_data)

    @override
    def _predict(self, context: None = None, predict_data: None = None) -> _PredictResponse[_ParameterValueType]:
        with _substitute_numpy_int():
            x = self.optimizer.ask()  # type: ignore

        x: List[_ParameterValueType] = x.tolist() if isinstance(x, np.ndarray) else x  # TODO Fix type hint
        prediction = {param.name: param.cast(value) for param, value in zip(self.params, x)}  # TODO Add type hint
        metadata = tuple(prediction.values())  # In the transformed normalized range
        return _PredictResponse(prediction, metadata)

    @override
    def _set_reward(self, tuning_request: _TuningRequest[_ParameterValueType]) -> None:
        reward = tuning_request.reward
        if reward is None:
            raise ValueError(f"reward cannot be None for {self.__class__.__name__}")

        # BO solves cost minimization, but the reward is passed assuming reward maximization.
        # So we negate the reward to make it compatible with BO (so that the reward acts as cost).
        reward = -reward
        x = tuning_request.metadata

        with _substitute_numpy_int():
            self.optimizer.tell(x, reward)

        self._iteration += 1

    @property
    @override
    def iteration(self) -> int:
        return self._iteration


@contextmanager
def _substitute_numpy_int() -> Generator[None, None, None]:
    """Temporarily reassign `np.int` to `int`, if the former is not defined."""
    if hasattr(np, "int"):
        yield
    else:
        # `np.int` was removed in NumPy 1.24 - https://numpy.org/devdocs/release/1.24.0-notes.html#expired-deprecations
        # `np.int` can be replaced with either `int` or `np.int_`.
        # Could have done this either in `BayesianOptimization`'s `__init__` or at the module level.
        # But this is to limit the scope of this change.
        np.int = int  # type: ignore

        try:
            yield
        finally:
            del np.int  # type: ignore


def _to_skopt(x: Union[Categorical, Integer, Real]) -> Union[space.Categorical, space.Integer, space.Real]:
    """Convert `oppertune.values.Value` to `skopt` equivalents.

    Note:
        Will have to use it with the `substitute_numpy_int` context manager if `np.int` is not defined
    """
    if isinstance(x, Categorical):
        return space.Categorical(categories=x.categories, name=x.name)

    if isinstance(x, Integer):
        return space.Integer(low=x.min, high=x.max, name=x.name)

    assert isinstance(x, Real)
    return space.Real(low=x.min, high=x.max, name=x.name)
