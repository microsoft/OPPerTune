from random import Random
from statistics import mean, median
from typing import Iterable, Union

import numpy as np
from numpy import allclose

from oppertune.algorithms.autoscope import AutoScope, AutoScopeFeature
from oppertune.core.types import Context
from oppertune.core.values import Real

from .helpers import root_mean_squared_error


class TestAutoScope:
    def test_onepoint(self) -> None:
        """Testing AutoScope with BlueFin leaves one point feedback."""

        def calculate_reward(prediction: Iterable[Union[int, float]], target: Iterable[Union[int, float]]) -> float:
            return -root_mean_squared_error(prediction, target)

        parameters = [
            Real("p1", val=0.5, min=0.0, max=1.0),
            Real("p2", val=0.3, min=0.0, max=1.0),
            Real("p3", val=0.1, min=0.0, max=1.0),
        ]

        jobtypes = ("0", "1", "2", "3", "4", "5", "6", "7")
        features = [
            AutoScopeFeature(name="jobtype", values=jobtypes),
        ]

        internal_weights = [
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, 1, -1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, -1, -1],
            [1, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, -1],
        ]

        optimal_params = {
            # jobtype: [p1, p2, p3]
            "0": [0.1, 0.3, 0.6],
            "1": [0.4, 0.5, 0.1],
            "2": [0.2, 0.3, 0.7],
            "3": [0.1, 0.7, 0.8],
            "4": [0.6, 0.7, 0.9],
            "5": [0.7, 0.8, 0.3],
            "6": [0.4, 0.6, 0.2],
            "7": [0.3, 0.2, 0.8],
        }

        expected_rewards = {
            "mean": -0.16468226743961345,
            "min": -0.565298492044158,
            "median": -0.14369665298740353,
            "p95": -0.04246640136190738,
            "p99": -0.025091036887005817,
            "max": -0.008458415453602675,
        }

        random_seed = 4
        tuning_instance = AutoScope(
            parameters=parameters,
            features=features,
            leaf_algorithm="bluefin",
            leaf_algorithm_args={
                "feedback": 1,
                "eta": 0.01,
                "delta": 0.1,
                "optimizer": "sgd",
                "random_seed": random_seed,
            },
            height=3,
            and_bias_is_learnable=False,
            over_param=None,
            eta=0.01,
            delta=0.1,
            optimizer="sgd",
            internal_weights=internal_weights,
            fix_internal_weights=True,
            random_seed=random_seed,
        )

        rewards = []
        iterations = 1000
        rng = Random(random_seed)
        for _ in range(iterations):
            jobtype = rng.choice(jobtypes)
            context = Context({"jobtype": jobtype})
            prediction, _ = tuning_instance.predict(context)
            reward = calculate_reward(prediction.values(), optimal_params[jobtype])
            tuning_instance.set_reward(reward, context.id)
            rewards.append(reward)

        assert allclose(mean(rewards), expected_rewards["mean"])
        assert allclose(min(rewards), expected_rewards["min"])
        assert allclose(median(rewards), expected_rewards["median"])
        assert allclose(np.percentile(rewards, 95), expected_rewards["p95"])
        assert allclose(np.percentile(rewards, 99), expected_rewards["p99"])
        assert allclose(max(rewards), expected_rewards["max"])

        # Internal weights are fixed and should not change
        trained_internal_weights = tuning_instance._predicate_layers[0].weight.detach().numpy()
        assert allclose(internal_weights, trained_internal_weights)

    def test_twopoint(self) -> None:
        """Testing AutoScope with BlueFin leaves two point feedback."""

        def calculate_reward(prediction: Iterable[Union[int, float]], target: Iterable[Union[int, float]]) -> float:
            return -root_mean_squared_error(prediction, target)

        parameters = [
            Real("p1", val=0.5, min=0.0, max=1.0),
            Real("p2", val=0.3, min=0.0, max=1.0),
            Real("p3", val=0.1, min=0.0, max=1.0),
        ]

        jobtypes = ("0", "1", "2", "3", "4", "5", "6", "7")
        features = [
            AutoScopeFeature(name="jobtype", values=jobtypes),
        ]

        internal_weights = [
            [0.0422, 0.0418, -0.2889, -0.2051, -0.3485, -0.326, 0.3485, 0.2921],
            [0.0838, 0.3354, -0.128, -0.2016, 0.3014, -0.0187, 0.0671, 0.209],
            [0.1863, -0.2024, -0.1367, -0.3263, 0.0156, -0.1268, 0.0759, 0.0164],
            [0.3014, 0.0305, 0.1772, 0.1116, 0.3137, -0.147, 0.2952, 0.2536],
            [-0.1411, 0.2167, 0.3269, -0.1999, -0.1379, 0.2419, 0.2851, 0.0933],
            [0.1086, 0.2619, 0.3098, -0.2236, -0.1454, 0.2977, 0.191, -0.1659],
            [-0.12, 0.3431, -0.2214, 0.2274, 0.2052, -0.3391, -0.3529, 0.1174],
        ]

        optimal_params = {
            # jobtype: [p1, p2, p3]
            "0": [0.1, 0.3, 0.6],
            "1": [0.4, 0.5, 0.1],
            "2": [0.2, 0.3, 0.7],
            "3": [0.1, 0.7, 0.8],
            "4": [0.6, 0.7, 0.9],
            "5": [0.7, 0.8, 0.3],
            "6": [0.4, 0.6, 0.2],
            "7": [0.3, 0.2, 0.8],
        }

        expected_rewards = {
            "mean": -0.2093698331219671,
            "min": -0.5547189657797458,
            "median": -0.19551992923666855,
            "p95": -0.0481508071320809,
            "p99": -0.03180158406715326,
            "max": -0.010721991711288844,
        }

        random_seed = 5
        tuning_instance = AutoScope(
            parameters=parameters,
            features=features,
            leaf_algorithm="bluefin",
            leaf_algorithm_args={
                "feedback": 2,
                "eta": 0.01,
                "delta": 0.1,
                "optimizer": "rmsprop",
                "random_seed": random_seed,
            },
            height=3,
            and_bias_is_learnable=False,
            over_param=None,
            eta=0.01,
            delta=0.1,
            optimizer="rmsprop",
            internal_weights=internal_weights,
            fix_internal_weights=False,
            random_seed=random_seed,
        )

        rewards = []
        iterations = 1000
        rng = Random(random_seed)
        for _ in range(iterations):
            jobtype = rng.choice(jobtypes)
            context = Context({"jobtype": jobtype})
            prediction, _ = tuning_instance.predict(context)
            reward = calculate_reward(prediction.values(), optimal_params[jobtype])
            tuning_instance.set_reward(reward, context.id)
            rewards.append(reward)

        assert allclose(mean(rewards), expected_rewards["mean"])
        assert allclose(min(rewards), expected_rewards["min"])
        assert allclose(median(rewards), expected_rewards["median"])
        assert allclose(np.percentile(rewards, 95), expected_rewards["p95"])
        assert allclose(np.percentile(rewards, 99), expected_rewards["p99"])
        assert allclose(max(rewards), expected_rewards["max"])

    def test_convergence_with_non_sequential_predicts(self) -> None:
        """Check if AutoScope converges with non sequential predicts."""

        def calculate_reward(prediction: Iterable[Union[int, float]], target: Iterable[Union[int, float]]) -> float:
            return -root_mean_squared_error(prediction, target)

        parameters = [
            Real("p1", val=0.5, min=0.0, max=1.0),
            Real("p2", val=0.3, min=0.0, max=1.0),
            Real("p3", val=0.1, min=0.0, max=1.0),
        ]

        jobtypes = ("0", "1", "2", "3", "4", "5", "6", "7")
        features = [
            AutoScopeFeature(name="jobtype", values=jobtypes),
        ]

        internal_weights = [
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, 1, -1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, -1, -1],
            [1, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, -1],
        ]

        optimal_params = {
            # jobtype: [p1, p2, p3]
            "0": [0.1, 0.3, 0.6],
            "1": [0.4, 0.5, 0.1],
            "2": [0.2, 0.3, 0.7],
            "3": [0.1, 0.7, 0.8],
            "4": [0.6, 0.7, 0.9],
            "5": [0.7, 0.8, 0.3],
            "6": [0.4, 0.6, 0.2],
            "7": [0.3, 0.2, 0.8],
        }

        for random_seed in range(6):  # Trying multiple seeds
            tuning_instance = AutoScope(
                parameters=parameters,
                features=features,
                leaf_algorithm="bluefin",
                leaf_algorithm_args={
                    "feedback": 2,
                    "eta": 0.01,
                    "delta": 0.1,
                    "optimizer": "sgd",
                    "random_seed": random_seed,
                },
                height=3,
                and_bias_is_learnable=False,
                over_param=None,
                eta=0.01,
                delta=0.1,
                optimizer="sgd",
                internal_weights=internal_weights,
                fix_internal_weights=True,
                random_seed=random_seed,
            )

            num_iterations = 1500
            rng = Random(random_seed)
            for _ in range(num_iterations):
                jobtype = rng.choice(jobtypes)
                context = Context({"jobtype": jobtype})
                prediction, _ = tuning_instance.predict(context)
                reward = calculate_reward(prediction.values(), optimal_params[jobtype])
                tuning_instance.set_reward(reward, context.id)

            for jobtype in jobtypes:
                context = Context({"jobtype": jobtype})
                prediction, _ = tuning_instance.predict(context)
                reward = calculate_reward(prediction.values(), optimal_params[jobtype])

                # Assuming converged if the reward is > -0.15.
                # This need not be true always especially if a jobtype does not receive enough samples.
                assert reward > -0.15
