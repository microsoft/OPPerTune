from typing import List, Mapping, Union

import numpy as np
from hyperopt import Trials, fmin, hp, tpe

from oppertune.algorithms.hopt import HOpt
from oppertune.core.values import Categorical, Integer, Real

from .helpers import squared_sum


class TestHOpt:
    def test_hopt(self) -> None:
        """Test if our wrapper over hyperopt imitates the library's behavior."""

        def calculate_loss(prediction: Mapping[str, Union[str, int, float]]) -> float:
            """Negative squared loss."""
            bias = {"f1": 0.0, "f2": -1.0}
            prediction_categorical: str = prediction["p1"]  # type: ignore
            prediction_numerical: List[float] = [prediction["p2"], prediction["p3"]]  # type: ignore
            target = [700, 1.0]
            return squared_sum(prediction_numerical, target) / (1000**2) + bias[prediction_categorical]

        # Run Hyperopt
        num_iterations = 100
        random_seed = 12345

        # http://hyperopt.github.io/hyperopt/getting-started/search_spaces/
        hopt_space = {
            "p1": hp.choice("p1", ("f1", "f2")),
            "p2": hp.quniform("p2", 100, 900, 100),
            "p3": hp.uniform("p3", 0.0, 10.0),
        }

        hopt_trials = Trials()

        fmin(
            fn=calculate_loss,
            space=hopt_space,
            algo=tpe.suggest,
            max_evals=num_iterations,
            trials=hopt_trials,
            rstate=np.random.default_rng(random_seed),
            show_progressbar=False,
        )

        # Run OPPerTune
        parameters = [
            Categorical("p1", val="f1", categories=("f1", "f2")),
            Integer("p2", val=100, min=100, max=900, step=100),
            Real("p3", val=5.0, min=0.0, max=10.0),
        ]

        tuning_instance = HOpt(parameters, algo="tpe", random_seed=random_seed)

        for i in range(num_iterations):
            prediction, _ = tuning_instance.predict()
            reward = -1 * calculate_loss(prediction)
            tuning_instance.set_reward(reward)

            hopt_vals = hopt_trials.trials[i]["misc"]["vals"]
            oppertune_vals = tuning_instance.fmin_args["trials"].trials[i]["misc"]["vals"]

            assert hopt_vals == oppertune_vals
