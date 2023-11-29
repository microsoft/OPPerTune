import unittest

import numpy as np

from oppertune import CategoricalValue, OPPerTune


class TestExpWeightsSlates(unittest.TestCase):
    def test_exp_weights_single_param(self):
        """Test working of exponential weights with a single categorical parameters."""

        seed = 4
        rng = np.random.default_rng(seed)

        def get_reward(pred) -> float:
            return float(rng.uniform(0, 1))

        parameters = (
            CategoricalValue(
                name="p1",
                initial_value=0,
                categories=(0, 1, 2),
            ),
        )

        tuner = OPPerTune(
            algorithm="exponential_weights_slates",
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
            for j in range(len(tuner.backend.p[0])):
                self.assertAlmostEqual(tuner.backend.p[0][j], _EXPECTED_P[i][j])
                self.assertAlmostEqual(tuner.backend.p_hat[0][j], _EXPECTED_P_HAT[i][j])
            self.assertAlmostEqual(tuner.backend.eta[0], _EXPECTED_ETA[i])
            self.assertAlmostEqual(tuner.backend.delta[0], _EXPECTED_DELTA[i])

            pred, _ = tuner.predict()

            reward = get_reward(pred)

            tuner.set_reward(reward)

    def test_exp_weights_multiple_params(self):
        """Test working of exponential weights with two categorical parameters."""

        seed = 4
        rng = np.random.default_rng(seed)

        def get_reward(pred) -> float:
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
                categories=(3, 4, 5),
            ),
        )

        tuner = OPPerTune(
            algorithm="exponential_weights_slates",
            parameters=parameters,
            algorithm_args=dict(random_seed=seed),
        )

        _EXPECTED_P = [
            [[0.33333333, 0.33333333, 0.33333333], [0.33333333, 0.33333333, 0.33333333]],
            [[0.73476185, 0.13261907, 0.13261907], [0.13261907, 0.73476185, 0.13261907]],
            [[0.65425648, 0.22765507, 0.11808845], [0.07885835, 0.84228330, 0.07885835]],
            [[0.46542115, 0.45057378, 0.08400507], [0.03152910, 0.93694180, 0.03152910]],
            [[0.45068102, 0.46797439, 0.08134459], [0.03418117, 0.93437607, 0.03144276]],
        ]
        _EXPECTED_P_HAT = [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.33333333, 0.33333333, 0.33333333], [0.33333333, 0.33333333, 0.33333333]],
            [[0.33333333, 0.33333333, 0.33333333], [0.33333333, 0.33333333, 0.33333333]],
            [[0.33333333, 0.33333333, 0.33333333], [0.33333333, 0.33333333, 0.33333333]],
            [[0.35102975, 0.34904058, 0.29992968], [0.29289923, 0.41420153, 0.29289923]],
        ]
        _EXPECTED_ETA = [
            [0.6051479953058617, 0.6051479953058617],
            [0.42790425110221986, 0.42790425110221986],
            [0.34938235798940165, 0.34938235798940165],
            [0.30257399765293086, 0.30257399765293086],
            [0.27063041079032607, 0.27063041079032607],
        ]
        _EXPECTED_DELTA = [
            [1, 1],
            [1, 1],
            [1, 1],
            [0.8660254037844386, 0.8660254037844386],
            [0.7745966692414834, 0.7745966692414834],
        ]

        num_iterations = 5

        for i in range(num_iterations):
            for j in range(len(tuner.backend.p[0])):
                self.assertAlmostEqual(tuner.backend.p[0][j], _EXPECTED_P[i][0][j])
                self.assertAlmostEqual(tuner.backend.p[1][j], _EXPECTED_P[i][1][j])

                self.assertAlmostEqual(tuner.backend.p_hat[0][j], _EXPECTED_P_HAT[i][0][j])
                self.assertAlmostEqual(tuner.backend.p_hat[1][j], _EXPECTED_P_HAT[i][1][j])

            self.assertAlmostEqual(tuner.backend.eta[0], _EXPECTED_ETA[i][0])
            self.assertAlmostEqual(tuner.backend.eta[1], _EXPECTED_ETA[i][1])

            self.assertAlmostEqual(tuner.backend.delta[0], _EXPECTED_DELTA[i][0])
            self.assertAlmostEqual(tuner.backend.delta[1], _EXPECTED_DELTA[i][1])

            pred, _ = tuner.predict()

            reward = get_reward(pred)

            tuner.set_reward(reward)
