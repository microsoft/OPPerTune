import unittest

import numpy as np

from oppertune import CategoricalValue, OPPerTune


class TestExpWeights(unittest.TestCase):
    def test_exp_weights_single_param(self):
        """Test working of exponential weights with a single categorical parameters."""

        seed = 4
        rng = np.random.default_rng(seed)

        def get_reward(pred) -> float:
            return rng.uniform(0, 1)

        parameters = (
            CategoricalValue(
                name="p1",
                initial_value=0,
                categories=(0, 1, 2),
            ),
        )

        tuner = OPPerTune(
            algorithm="exponential_weights",
            parameters=parameters,
            algorithm_args=dict(random_seed=seed),
        )

        _EXPECTED_P = [
            [0.33333333, 0.33333333, 0.33333333],
            [0.73476185, 0.13261907, 0.13261907],
            [0.65425648, 0.22765507, 0.11808845],
            [0.46542115, 0.45057378, 0.08400507],
            [0.46212271, 0.44738057, 0.09049672],
        ]
        _EXPECTED_P_HAT = [
            [0.0, 0.0, 0.0],
            [0.33333333, 0.33333333, 0.33333333],
            [0.33333333, 0.33333333, 0.33333333],
            [0.33333333, 0.33333333, 0.33333333],
            [0.35102975, 0.34904058, 0.29992968],
        ]
        _EXPECTED_ETA = [
            0.6051479953058617,
            0.42790425110221986,
            0.34938235798940165,
            0.30257399765293086,
            0.27063041079032607,
        ]
        _EXPECTED_DELTA = [1, 1, 1, 0.8660254037844386, 0.7745966692414834]

        num_iterations = 5

        for i in range(num_iterations):
            for j in range(len(tuner.backend.p)):
                self.assertAlmostEqual(tuner.backend.p[j], _EXPECTED_P[i][j])
                self.assertAlmostEqual(tuner.backend.p_hat[j], _EXPECTED_P_HAT[i][j])
            self.assertAlmostEqual(tuner.backend.eta, _EXPECTED_ETA[i])
            self.assertAlmostEqual(tuner.backend.delta, _EXPECTED_DELTA[i])

            pred, _ = tuner.predict()

            reward = get_reward(pred)

            tuner.set_reward(reward)

    def test_exp_weights_multiple_params(self):
        """Test working of exponential weights with two categorical parameters."""

        seed = 4
        rng = np.random.default_rng(seed)

        def get_reward(pred, rng) -> float:
            return float(rng.uniform(0, 1))

        parameters = (
            CategoricalValue(
                name="p1",
                initial_value=0,
                categories=(0, 1, 2),
            ),
            CategoricalValue(
                name="p2",
                initial_value=3,
                categories=(3, 4),
            ),
        )

        tuner = OPPerTune(
            algorithm="exponential_weights",
            parameters=parameters,
            algorithm_args=dict(random_seed=seed),
        )

        _EXPECTED_P = [
            [0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667],
            [0.81497378, 0.03700524, 0.03700524, 0.03700524, 0.03700524, 0.03700524],
            [0.75176033, 0.11169994, 0.03413493, 0.03413493, 0.03413493, 0.03413493],
            [0.63572007, 0.09445815, 0.02886593, 0.18322399, 0.02886593, 0.02886593],
            [0.62732323, 0.1064189, 0.02848466, 0.1808039, 0.02848466, 0.02848466],
        ]
        _EXPECTED_P_HAT = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667],
            [0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667],
            [0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667],
            [0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667],
        ]
        _EXPECTED_ETA = [
            0.5464673624331794,
            0.3864107776736279,
            0.3155030788041409,
            0.2732336812165897,
            0.2443876339771208,
        ]
        _EXPECTED_DELTA = [1, 1, 1, 1, 1]

        num_iterations = 5

        for i in range(num_iterations):
            for j in range(len(tuner.backend.p)):
                self.assertAlmostEqual(tuner.backend.p[j], _EXPECTED_P[i][j])
                self.assertAlmostEqual(tuner.backend.p_hat[j], _EXPECTED_P_HAT[i][j])

            self.assertAlmostEqual(tuner.backend.eta, _EXPECTED_ETA[i])
            self.assertAlmostEqual(tuner.backend.delta, _EXPECTED_DELTA[i])

            pred, _ = tuner.predict()

            reward = get_reward(pred, rng)

            tuner.set_reward(reward)
