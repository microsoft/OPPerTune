from statistics import mean, median
from typing import Iterable, Union

import numpy as np
import pytest
from numpy import allclose

from oppertune.algorithms.bluefin import BlueFin
from oppertune.core.values import Integer, Real

from .helpers import absolute_error, root_mean_squared_error, squared_sum


class TestBluefin:
    def test_getters(self) -> None:
        """Test for getters."""
        parameters = [
            Real("p1", val=5.0, min=1.0, max=10.0),
        ]

        feedback = 1
        eta = 0.1
        delta = 0.01
        random_seed = 12345
        eta_decay_rate = 0.1

        tuning_instance = BlueFin(
            parameters,
            feedback=feedback,
            eta=eta,
            delta=delta,
            random_seed=random_seed,
            eta_decay_rate=eta_decay_rate,
        )

        assert tuning_instance.feedback == feedback
        assert tuning_instance.eta == eta
        assert tuning_instance.delta == delta
        assert tuning_instance._random_seed == random_seed
        assert tuning_instance.eta_decay_rate == eta_decay_rate

    def test_duplicate_params(self) -> None:
        """Test for same parameter name."""

        def create_invalid_tuning_instance() -> BlueFin:
            return BlueFin(
                parameters=[
                    Real("p1", val=0.9, min=0.1, max=1.0),
                    Real("p1", val=0.5, min=0.2, max=1.0),
                ]
            )

        pytest.raises(ValueError, create_invalid_tuning_instance)

    def test_features_to_predict(self) -> None:
        """Bluefin ignores any additional arguments passed to it during predict."""
        parameters = [
            Real("p1", val=4.0, min=1.0, max=10.0),
            Real("p2", val=4.0, min=1.0, max=10.0),
        ]

        tuning_instance = BlueFin(parameters)

        pytest.raises(
            TypeError,
            tuning_instance.predict,
            features=[1, 0, 0],
            tags={"job_type": 1, "data_type": 2},
        )

    def test_consecutive_predict(self) -> None:
        """Consecutive predict calls should return the same values."""
        parameters = [
            Real("p1", val=4.0, min=1.0, max=10.0),
            Real("p2", val=4.0, min=1.0, max=10.0),
        ]

        tuning_instance = BlueFin(parameters)

        pred1, _ = tuning_instance.predict()
        pred2, _ = tuning_instance.predict()

        assert pred1 == pred2

    def test_onepoint(self) -> None:
        """Testing onepoint feedback."""

        def calculate_reward(prediction: Iterable[Union[int, float]], target: Iterable[Union[int, float]]) -> float:
            return -squared_sum(prediction, target) / (100**2)

        parameters = [
            Real("p1", val=50.0, min=1.0, max=100.0),
            Real("p2", val=20.0, min=1.0, max=100.0),
            Real("p3", val=30.0, min=1.0, max=100.0),
            Integer("p4", val=70, min=1, max=100),
            Integer("p5", val=10, min=1, max=100),
        ]

        target = [10.0, 50.0, 70.0, 20, 80]

        expected_rewards = {
            "mean": -0.03699159848296742,
            "min": -1.3842386961672057,
            "median": -0.011287915220024269,
            "p95": -0.006658190445350101,
            "p99": -0.005095832210173069,
            "max": -0.001104576566414092,
        }

        tuning_instance = BlueFin(
            parameters,
            feedback=1,
            optimizer="sgd",
            eta=0.01,
            delta=0.1,
            random_seed=2,
        )
        num_iterations = 1000

        rewards = []
        for _ in range(num_iterations):
            prediction, _ = tuning_instance.predict()
            reward = calculate_reward(prediction.values(), target)
            rewards.append(reward)
            tuning_instance.set_reward(reward)

        assert allclose(mean(rewards), expected_rewards["mean"])
        assert allclose(min(rewards), expected_rewards["min"])
        assert allclose(median(rewards), expected_rewards["median"])
        assert allclose(np.percentile(rewards, 95), expected_rewards["p95"])
        assert allclose(np.percentile(rewards, 99), expected_rewards["p99"])
        assert allclose(max(rewards), expected_rewards["max"])

    def test_twopoint(self) -> None:
        """Testing twopoint feedback."""

        def calculate_reward(prediction: Iterable[Union[int, float]], target: Iterable[Union[int, float]]) -> float:
            return -absolute_error(prediction, target) / 1000

        parameters = [
            Real("p1", val=400.0, min=1.0, max=1000.0),
            Real("p2", val=400.0, min=1.0, max=1000.0),
        ]

        target = [200.0, 800.0]

        expected_rewards = {
            "mean": -0.07657255867260056,
            "min": -0.6639602790056434,
            "median": -0.022036472387340637,
            "p95": -0.014625555310080925,
            "p99": -0.012693946801577387,
            "max": -0.01088593460415541,
        }

        tuning_instance = BlueFin(
            parameters,
            feedback=2,
            eta=0.0025,
            delta=0.05,
            optimizer="sgd",
            eta_decay_rate=0.03,
            random_seed=2,
        )
        num_iterations = 1000

        rewards = []
        for _ in range(num_iterations):
            prediction, _ = tuning_instance.predict()
            reward = calculate_reward(prediction.values(), target)
            rewards.append(reward)
            tuning_instance.set_reward(reward)

        assert allclose(mean(rewards), expected_rewards["mean"])
        assert allclose(median(rewards), expected_rewards["median"])
        assert allclose(min(rewards), expected_rewards["min"])
        assert allclose(max(rewards), expected_rewards["max"])
        assert allclose(np.percentile(rewards, 95), expected_rewards["p95"])
        assert allclose(np.percentile(rewards, 99), expected_rewards["p99"])

    def test_twopoint_zero_reward(self) -> None:
        """Test if zero division error occurs when reward is zero."""

        def calculate_reward() -> float:
            return 0

        parameters = [
            Integer("p1", val=4, min=1, max=10),
            Integer("p2", val=4, min=1, max=10),
        ]

        tuning_instance = BlueFin(
            parameters,
            feedback=2,
            eta=0.01,
            delta=0.1,
            optimizer="sgd",
            random_seed=2,
        )
        num_iterations = 5

        for _ in range(num_iterations):
            tuning_instance.predict()
            reward = calculate_reward()
            tuning_instance.set_reward(reward)

    def test_step_size(self) -> None:
        """Testing Step size."""

        def calculate_reward(prediction: Iterable[Union[int, float]], target: Iterable[Union[int, float]]) -> float:
            return -squared_sum(prediction, target) / (1000**2)

        expected_rewards = {
            "mean": -0.010543416,
            "min": -0.000036,
            "median": -0.360016,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

        parameters = [
            Integer("p1", val=5, min=0, max=10),
            Real("p2", val=100.0, min=100.0, max=900.0, step=100.0),
        ]

        target = [1, 700.0]

        tuning_instance = BlueFin(
            parameters,
            feedback=2,
            eta=0.01,
            delta=0.1,
            optimizer="sgd",
            random_seed=4,
        )
        num_iterations = 1000

        rewards = []
        for _ in range(num_iterations):
            prediction, _ = tuning_instance.predict()
            reward = calculate_reward(prediction.values(), target)
            rewards.append(reward)
            tuning_instance.set_reward(reward)

        expected_rewards = {
            "mean": -0.010543416,
            "min": -0.360016,
            "median": -0.000036,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

        assert allclose(mean(rewards), expected_rewards["mean"])
        assert allclose(min(rewards), expected_rewards["min"])
        assert allclose(median(rewards), expected_rewards["median"])
        assert allclose(np.percentile(rewards, 95), expected_rewards["p95"])
        assert allclose(np.percentile(rewards, 99), expected_rewards["p99"])
        assert allclose(max(rewards), expected_rewards["max"])

    def test_zero_eta(self) -> None:
        """Zero eta should raise an error."""
        parameters = [
            Real("p1", val=0.9, min=0.0, max=1.0),
            Real("p2", val=0.5, min=0.0, max=1.0),
        ]

        def create_invalid_tuning_instance() -> BlueFin:
            return BlueFin(
                parameters,
                feedback=2,
                eta=0.0,
                delta=0.01,
                optimizer="sgd",
                random_seed=4,
            )

        pytest.raises(ValueError, create_invalid_tuning_instance)

    def test_zero_delta(self) -> None:
        """Zero delta should raise an error."""
        parameters = [
            Real("p1", val=0.9, min=0.0, max=1.0),
            Real("p2", val=0.5, min=0.0, max=1.0),
        ]

        def create_invalid_tuning_instance() -> BlueFin:
            return BlueFin(
                parameters,
                feedback=2,
                eta=0.1,
                delta=0.0,
                optimizer="sgd",
                random_seed=4,
            )

        pytest.raises(ValueError, create_invalid_tuning_instance)

    def test_rmsprop(self) -> None:
        """Testing the RMSProp optimizer."""

        def calculate_reward(prediction: Iterable[Union[int, float]], target: Iterable[Union[int, float]]) -> float:
            return -root_mean_squared_error(prediction, target)

        expected_param_values = [
            {"p1": 0.8373197466431326, "p2": 0.29265432476549247, "p3": 0.19985661183211623},
            {"p1": 0.9099279533568674, "p2": 0.3121174752345075, "p3": 0.01452114816788376},
            {"p1": 0.8108889066001379, "p2": 0.10958923002700288, "p3": 0.2068947079772521},
            {"p1": 0.7363588162667183, "p2": 0.2951826552788627, "p3": 0.20748304306428142},
            {"p1": 0.6821116380133826, "p2": 0.3096946119254001, "p3": 0.11450100546949132},
            {"p1": 0.7541382555969532, "p2": 0.29252372421801603, "p3": 0.3002893984729577},
            {"p1": 0.7586115959488672, "p2": 0.31123366067966113, "p3": 0.3356665647943705},
            {"p1": 0.7286030218256602, "p2": 0.2820183282830014, "p3": 0.14010084736914108},
            {"p1": 0.7816596725272298, "p2": 0.34166137271505076, "p3": 0.19431456668459235},
            {"p1": 0.7423113706828501, "p2": 0.27821763254836546, "p3": 0.3798586395503827},
        ]

        parameters = [
            Real("p1", val=0.87362385, min=0.0, max=1.0),
            Real("p2", val=0.30238590, min=0.0, max=1.0),
            Real("p3", val=0.10718888, min=0.0, max=1.0),
        ]

        target = [0.1, 0.3, 0.6]

        tuning_instance = BlueFin(
            parameters,
            feedback=2,
            eta=0.01,
            delta=0.1,
            optimizer="rmsprop",
            random_seed=4,
        )

        num_iterations = 10

        param_values = []
        rewards = []
        for _ in range(num_iterations):
            prediction, _ = tuning_instance.predict()
            reward = calculate_reward(prediction.values(), target)
            rewards.append(reward)
            param_values.append(prediction)
            tuning_instance.set_reward(reward)

        for idx, values in enumerate(param_values):
            for param in parameters:
                assert allclose(values[param.name], expected_param_values[idx][param.name])

        expected_rewards = {
            "mean": -0.45383194460759685,
            "min": -0.5770371474979047,
            "median": -0.44373576396852016,
            "p95": -0.40012148002065384,
            "p99": -0.3937972371008036,
            "max": -0.39221617637084105,
        }

        assert allclose(mean(rewards), expected_rewards["mean"])
        assert allclose(min(rewards), expected_rewards["min"])
        assert allclose(median(rewards), expected_rewards["median"])
        assert allclose(np.percentile(rewards, 95), expected_rewards["p95"])
        assert allclose(np.percentile(rewards, 99), expected_rewards["p99"])
        assert allclose(max(rewards), expected_rewards["max"])
