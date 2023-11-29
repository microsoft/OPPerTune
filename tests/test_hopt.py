import os
import unittest

import numpy as np
from hyperopt import Trials, fmin, hp, tpe

from oppertune import CategoricalValue, ContinuousValue, DiscreteValue, OPPerTune, dump, load


class TestHOpt(unittest.TestCase):
    def test_hopt(self):
        """Test if our wrapper over hyperopt imitates the library's behavior."""

        def get_loss(pred) -> float:
            """Negative squared loss."""
            bias = {
                "f1": 0.0,
                "f2": -1.0,
            }
            pred_num = np.asarray([pred["p1"], pred["p2"]])
            pred_cat = pred["p3"]
            target = np.array([1, 700])
            return np.square(pred_num - target).sum() / (1000**2) + bias[pred_cat]

        # Run Hyperopt
        num_iterations = 100
        seed = 4

        # http://hyperopt.github.io/hyperopt/getting-started/search_spaces/
        hopt_space = {
            "p1": hp.uniform("p1", 0.0, 10.0),
            "p2": hp.quniform("p2", 100, 900, 100),
            "p3": hp.choice("p3", ("f1", "f2")),
        }

        hopt_trials = Trials()

        fmin(
            fn=get_loss,
            space=hopt_space,
            algo=tpe.suggest,
            max_evals=num_iterations,
            trials=hopt_trials,
            rstate=np.random.default_rng(seed),
            show_progressbar=False,
        )

        # Run OPPerTune
        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=5.0,
                lb=0.0,
                ub=10.0,
            ),
            DiscreteValue(
                name="p2",
                initial_value=100,
                lb=100,
                ub=900,
                step_size=100,
            ),
            CategoricalValue(
                name="p3",
                initial_value="f1",
                categories=("f1", "f2"),
            ),
        )

        tuner = OPPerTune(
            algorithm="hopt",
            parameters=parameters,
            algorithm_args=dict(
                algo="tpe",
                random_seed=seed,
            ),
        )

        for i in range(num_iterations):
            pred, _metadata = tuner.predict()
            reward = -1 * get_loss(pred)
            tuner.set_reward(reward, metadata=_metadata)

            hopt_vals = hopt_trials.trials[i]["misc"]["vals"]
            oppertune_vals = tuner.backend.fmin_args["trials"].trials[i]["misc"]["vals"]

            self.assertDictEqual(hopt_vals, oppertune_vals)

    def test_pickling(self):
        """Test if the tuner can be pickled."""

        def get_loss(pred) -> float:
            """Negative squared loss."""
            bias = {
                "f1": 0.0,
                "f2": -1.0,
            }
            pred_num = np.asarray([pred["p1"], pred["p2"]])
            pred_cat = pred["p3"]
            target = np.array([1, 700])
            return np.square(pred_num - target).sum() / (1000**2) + bias[pred_cat]

        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=5.0,
                lb=0.0,
                ub=10.0,
            ),
            DiscreteValue(
                name="p2",
                initial_value=100,
                lb=100,
                ub=900,
                step_size=100,
            ),
            CategoricalValue(
                name="p3",
                initial_value="f1",
                categories=("f1", "f2"),
            ),
        )

        num_iterations = 100
        seed = 4
        tuner = OPPerTune(
            algorithm="hopt",
            parameters=parameters,
            algorithm_args=dict(
                algo="tpe",
                random_seed=seed,
            ),
        )

        fname = "tuner.joblib"
        dump(tuner, fname)

        for _ in range(num_iterations):
            pred, _metadata = tuner.predict()
            reward = -1 * get_loss(pred)
            tuner.set_reward(reward, metadata=_metadata)

            tuner_obj = load(fname)
            pred_obj, _metadata_obj = tuner_obj.predict()
            reward_obj = -1 * get_loss(pred_obj)
            tuner_obj.set_reward(reward_obj, metadata=_metadata_obj)

            self.assertDictEqual(pred, pred_obj)
            self.assertEqual(reward, reward_obj)
            self.assertEqual(_metadata, _metadata_obj)

            dump(tuner_obj, fname)
        os.remove(fname)
