from typing import Iterable, Optional, Union

import numpy as np
from hyperopt import Trials, anneal, atpe, fmin, hp, rand, tpe

from ...values import CategoricalValue, ContinuousValue, DiscreteValue
from ..base import AlgorithmBackend, PredictResponse

__all__ = ("HOpt",)


class HOpt(AlgorithmBackend):
    _algorithm_map = {
        "tpe": tpe.suggest,
        "rand": rand.suggest,
        "anneal": anneal.suggest,
        "atpe": atpe.suggest,
    }

    def __init__(
        self,
        parameters: Iterable[Union[ContinuousValue, DiscreteValue, CategoricalValue]],
        algo: str = "tpe",
        fmin_args: Optional[dict] = None,
        random_seed=None,
    ):
        """
        Args:
            fmin_args: arguments to fmin. Refer https://github.com/hyperopt/hyperopt/wiki/FMin for more details.
                Do not give "fn", "space", "trials", "rstate" or "show_progress_bar" as arguments since they will be
                overwritten.
        """

        self.params = tuple(parameters)

        assert algo in HOpt._algorithm_map, f"Unknown algorithm {algo}"

        self.fmin_args = fmin_args if fmin_args is not None else {}
        self.fmin_args["algo"] = HOpt._algorithm_map[algo]
        self.fmin_args["fn"] = self.dummy_objective
        self.fmin_args["trials"] = Trials()  # Create a trials object to store results
        self.fmin_args["space"] = HOpt.get_space(parameters)
        self.fmin_args["rstate"] = np.random.default_rng(random_seed)
        self.fmin_args["show_progressbar"] = False  # Suppress the progress bar

        self._round = 1

    def dummy_objective(self, values: dict):
        return -1.0

    def format(self, values: dict):
        return {param.name: param.cast(value[0]) for param, value in zip(self.params, values.values())}

    def predict(self):
        fmin(**self.fmin_args, max_evals=self._round)  # Run one round of optimization
        values = self.last_trial["misc"]["vals"]  # Get the last set of parameters
        parameters = self.format(values)
        return PredictResponse(parameters)

    def set_reward(self, reward: Union[float, int], metadata=None):
        reward = -reward
        self.last_trial["result"]["loss"] = reward  # Store the reward to be used in the next predict call
        self._round += 1

    @property
    def last_trial(self):
        return self.fmin_args["trials"].trials[-1]

    @property
    def round(self):
        return self._round

    @staticmethod
    def get_space(parameters):
        space = {}
        for param in parameters:
            if isinstance(param, ContinuousValue):
                if param.step_size is None:
                    space[param.name] = hp.uniform(param.name, param.lb, param.ub)
                else:
                    space[param.name] = hp.quniform(param.name, param.lb, param.ub, param.step_size)
            elif isinstance(param, DiscreteValue):
                space[param.name] = hp.quniform(param.name, param.lb, param.ub, param.step_size or 1)
            elif isinstance(param, CategoricalValue):
                space[param.name] = hp.choice(param.name, param.categories)

        return space
