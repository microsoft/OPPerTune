from typing import Iterable, Mapping, Union

import numpy as np
import pytest

from oppertune.algorithms.hybrid_solver import HybridSolver
from oppertune.core.values import Categorical, Real

from .helpers import absolute_error, root_mean_squared_error


class TestHybridSolver:
    def test_hybrid_solver(self) -> None:
        """Test working of hybrid solver."""

        def f1(prediction: Mapping[str, Union[str, int, float]]) -> float:
            optimal_params = {"p1": 3, "p2": 5, "p3": 7, "p4": 9, "p5": 1, "p6": 3}
            _pred = [prediction[k] for k in optimal_params]

            optimal_arr = np.asarray(list(optimal_params.values()), dtype=float)
            pred_arr = np.asarray(_pred)

            return -1 * root_mean_squared_error(pred_arr, optimal_arr)

        def f2(prediction: Mapping[str, Union[str, int, float]]) -> float:
            optimal_params = {"p1": 9, "p2": 1, "p3": 3, "p4": 5, "p5": 7, "p6": 1}
            _pred = [prediction[k] for k in optimal_params]

            optimal_arr = np.asarray(list(optimal_params.values()))
            pred_arr = np.asarray(_pred)

            return -1 * root_mean_squared_error(pred_arr, optimal_arr)

        def calculate_reward(prediction: Mapping[str, Union[str, int, float]]) -> float:
            if prediction["category"] == "f1":
                return f1(prediction)
            elif prediction["category"] == "f2":
                return f2(prediction)
            else:
                raise Exception("Invalid value for category")

        parameters = [
            Real("p1", val=1, min=0.0, max=10.0),
            Real("p2", val=3, min=0.0, max=10.0),
            Real("p3", val=5, min=0.0, max=10.0),
            Real("p4", val=7, min=0.0, max=10.0),
            Real("p5", val=9, min=0.0, max=10.0),
            Real("p6", val=2, min=0.0, max=10.0),
            Categorical("category", val="f1", categories=("f1", "f2")),
        ]

        expected_rewards = [-3.396297354084446, -5.351881890813031, -4.4886629741537885]
        tuning_instance = HybridSolver(
            parameters,
            categorical_algorithm="exponential_weights_slates",
            categorical_algorithm_args={
                "random_seed": 4,
            },
            numerical_algorithm="bluefin",
            numerical_algorithm_args={
                "feedback": 1,
                "optimizer": "sgd",
                "random_seed": 4,
            },
        )

        num_iterations = 3

        for i in range(num_iterations):
            prediction, _ = tuning_instance.predict()
            reward = calculate_reward(prediction)
            tuning_instance.set_reward(reward)
            assert reward == pytest.approx(expected_rewards[i])

    def test_only_numerical_params(self) -> None:
        def calculate_reward(prediction: Iterable[Union[int, float]], target: Iterable[Union[int, float]]) -> float:
            """Absolute Loss."""
            return -absolute_error(prediction, target) / 1000

        parameters = [
            Real("p1", val=400.0, min=1.0, max=1000.0),
            Real("p2", val=400.0, min=1.0, max=1000.0),
        ]

        target = [200, 800]

        tuning_instance = HybridSolver[float](
            parameters,
            categorical_algorithm="exponential_weights_slates",
            numerical_algorithm="bluefin",
            numerical_algorithm_args={
                "feedback": 2,
                "eta": 0.0025,
                "delta": 0.05,
                "optimizer": "sgd",
                "eta_decay_rate": 0.03,
                "random_seed": 2,
            },
        )

        num_iterations = 100

        for _ in range(num_iterations):  # Should run without errors
            prediction, _ = tuning_instance.predict()
            reward = calculate_reward(prediction.values(), target)
            tuning_instance.set_reward(reward)

    def test_only_categorical_params(self) -> None:
        parameters = [
            Categorical("p1", val="a", categories=("a", "b")),
            Categorical("p2", val="e", categories=("c", "d", "e", "f")),
        ]

        tuning_instance = HybridSolver[str](
            parameters,
            categorical_algorithm="exponential_weights_slates",
            numerical_algorithm="bluefin",
            numerical_algorithm_args={
                "feedback": 2,
                "eta": 0.01,
                "delta": 0.1,
                "random_seed": 4,
            },
        )

        num_iterations = 100

        for _ in range(num_iterations):  # Should run without erroring
            tuning_instance.predict()
            tuning_instance.set_reward(1.0)  # Dummy reward
