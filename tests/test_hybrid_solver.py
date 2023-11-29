import unittest

import numpy as np

from oppertune import CategoricalValue, ContinuousValue, OPPerTune


class TestHybridSolver(unittest.TestCase):
    def test_hybrid_solver(self):
        """Test working of hybrid solver."""

        def rmse(p, t):
            return np.sqrt(np.mean(np.square(p - t)))

        def f1(pred: dict):
            optimal_params = {"p1": 3, "p2": 5, "p3": 7, "p4": 9, "p5": 1, "p6": 3}
            _pred = [pred[k] for k in optimal_params]

            optimal_arr = np.asarray(list(optimal_params.values()), dtype=float)
            pred_arr = np.asarray(_pred)

            return -1 * rmse(pred_arr, optimal_arr)

        def f2(pred: dict):
            optimal_params = {"p1": 9, "p2": 1, "p3": 3, "p4": 5, "p5": 7, "p6": 1}
            _pred = [pred[k] for k in optimal_params]

            optimal_arr = np.asarray(list(optimal_params.values()))
            pred_arr = np.asarray(_pred)

            return -1 * rmse(pred_arr, optimal_arr)

        def get_reward(pred):
            if pred["category"] == "f1":
                return f1(pred)

            elif pred["category"] == "f2":
                return f2(pred)

            else:
                raise Exception("Invalid value for category")

        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=1,
                lb=0.0,
                ub=10.0,
            ),
            ContinuousValue(
                name="p2",
                initial_value=3,
                lb=0.0,
                ub=10.0,
            ),
            ContinuousValue(
                name="p3",
                initial_value=5,
                lb=0.0,
                ub=10.0,
            ),
            ContinuousValue(
                name="p4",
                initial_value=7,
                lb=0.0,
                ub=10.0,
            ),
            ContinuousValue(
                name="p5",
                initial_value=9,
                lb=0.0,
                ub=10.0,
            ),
            ContinuousValue(
                name="p6",
                initial_value=2,
                lb=0.0,
                ub=10.0,
            ),
            CategoricalValue(
                name="category",
                initial_value="f1",
                categories=("f1", "f2"),
            ),
        )

        _EXPECTED_REWARDS = [-3.396297354084446, -5.351881890813031, -4.4886629741537885]
        tuner = OPPerTune(
            algorithm="hybrid_solver",
            parameters=parameters,
            algorithm_args=dict(
                numerical_solver="bluefin",
                numerical_solver_args=dict(random_seed=4),
                categorical_solver="exponential_weights_slates",
                categorical_solver_args=dict(
                    random_seed=4,
                ),
            ),
        )

        num_iterations = 3

        for i in range(num_iterations):
            pred, _ = tuner.predict()

            reward = get_reward(pred)

            tuner.set_reward(reward)
            self.assertAlmostEqual(reward, _EXPECTED_REWARDS[i])

    def test_only_numerical_params(self):
        def get_reward(prediction: dict):
            """Absolute Loss."""
            pred = np.array(list(prediction.values()))
            target = np.array([200, 800])
            return -np.abs(pred - target).sum() / 1000

        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=400.0,
                lb=1.0,
                ub=1000.0,
            ),
            ContinuousValue(
                name="p2",
                initial_value=400.0,
                lb=1.0,
                ub=1000.0,
            ),
        )

        tuner = OPPerTune(
            algorithm="hybrid_solver",
            parameters=parameters,
            algorithm_args=dict(
                numerical_solver="bluefin",
                categorical_solver="exponential_weights_slates",
                numerical_solver_args=dict(
                    feedback=2,
                    eta=0.0025,
                    delta=0.05,
                    optimizer="sgd",
                    normalize=True,
                    eta_decay_rate=0.03,
                    random_seed=2,
                ),
            ),
        )

        num_iterations = 5

        for _ in range(num_iterations):  # Should run without errors
            pred, _ = tuner.predict()
            reward = get_reward(pred)
            tuner.set_reward(reward)

    def test_only_categorical_params(self):
        parameters = (
            CategoricalValue(
                name="p1",
                type="categorical",
                initial_value="t",
                categories=("t", "f"),
            ),
            CategoricalValue(
                name="p2",
                type="categorical",
                initial_value="t",
                categories=("t", "f"),
            ),
        )

        tuner = OPPerTune(
            algorithm="hybrid_solver",
            parameters=parameters,
            algorithm_args=dict(
                numerical_solver="bluefin",
                categorical_solver="exponential_weights_slates",
                numerical_solver_args=dict(feedback=2, eta=0.01, delta=0.1, random_seed=4),
            ),
        )

        num_iterations = 5

        for _ in range(num_iterations):  # Should run without erroring
            tuner.predict()
            tuner.set_reward(1.0)  # Dummy reward
