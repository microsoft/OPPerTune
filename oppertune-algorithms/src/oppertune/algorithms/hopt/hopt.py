"""The HOpt algorithm."""

from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
from hyperopt import Trials, anneal, atpe, fmin, hp, rand, tpe
from hyperopt.pyll.base import Apply
from typing_extensions import TypeVar, override

from oppertune.core.types import PredictResponse
from oppertune.core.values import Categorical, Integer, Real

from ..base import Algorithm, _PredictResponse, _TuningRequest

__all__ = ("HOpt",)

ALGORITHMS = {
    "tpe": tpe.suggest,
    "rand": rand.suggest,
    "anneal": anneal.suggest,
    "atpe": atpe.suggest,
}

_ParameterValueType = TypeVar("_ParameterValueType", bound=Union[str, int, float], default=Union[str, int, float])


class HOpt(Algorithm[_ParameterValueType]):
    class Meta:
        supported_parameter_types = (Categorical, Integer, Real)
        requires_untransformed_parameters = True
        supports_context = False
        supports_single_reward = True
        supports_sequence_of_rewards = False

    def __init__(
        self,
        parameters: Iterable[Union[Categorical, Integer, Real]],
        algo: str = "tpe",
        fmin_args: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
    ):
        """Args:
        fmin_args: arguments to fmin. Refer https://github.com/hyperopt/hyperopt/wiki/FMin for more details.
            Do not give "fn", "space", "trials", "rstate" or "show_progress_bar" as arguments since they will be
            overwritten.
        """
        super().__init__(parameters, random_seed=random_seed)

        assert algo in ALGORITHMS, f"Unknown algorithm {algo}"
        self.fmin_args = fmin_args or {}
        self.fmin_args["algo"] = ALGORITHMS[algo]
        self.fmin_args["fn"] = self.dummy_objective
        self.fmin_args["trials"] = Trials()  # Create a trials object to store results
        self.fmin_args["space"] = HOpt.get_space(self.params)
        self.fmin_args["rstate"] = np.random.default_rng(self._random_seed)
        self.fmin_args["show_progressbar"] = False  # Suppress the progress bar

        self._iteration = 1

    def dummy_objective(self, values: dict) -> float:
        return -1.0

    @override
    def predict(self, context: None = None, predict_data: None = None) -> PredictResponse[_ParameterValueType]:
        return super().predict(context, predict_data)

    @override
    def _predict(self, context: None = None, predict_data: None = None) -> _PredictResponse[_ParameterValueType]:
        fmin(**self.fmin_args, max_evals=self._iteration)  # Run one round of optimization
        values: dict = self.last_trial["misc"]["vals"]  # Get the last set of parameters
        prediction = {param.name: param.cast(value[0]) for param, value in zip(self.params, values.values())}
        return _PredictResponse(prediction)

    @override
    def _set_reward(self, tuning_request: _TuningRequest[_ParameterValueType]) -> None:
        reward = tuning_request.reward
        if reward is None:
            raise ValueError(f"reward cannot be None for {self.__class__.__name__}")

        self.last_trial["result"]["loss"] = -1 * reward  # Store the loss to be used in the next predict call
        self._iteration += 1

    @staticmethod
    def get_space(parameters: Iterable[Union[Categorical, Integer, Real]]) -> Dict[str, Apply]:
        space: Dict[str, Apply] = {}
        for param in parameters:
            if isinstance(param, Categorical):
                space[param.name] = hp.choice(param.name, param.categories)
            elif isinstance(param, Integer):
                space[param.name] = hp.quniform(param.name, param.min, param.max, param.step or 1)  # TODO Not needed?
            elif isinstance(param, Real):
                if param.step is None:
                    space[param.name] = hp.uniform(param.name, param.min, param.max)
                else:
                    space[param.name] = hp.quniform(param.name, param.min, param.max, param.step)

        return space

    @property
    def last_trial(self):
        return self.fmin_args["trials"].trials[-1]

    @property
    @override
    def iteration(self) -> int:
        return self._iteration
