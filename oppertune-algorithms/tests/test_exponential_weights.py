from random import Random
from typing import Mapping

from numpy import allclose

from oppertune.algorithms.exponential_weights import ExponentialWeights
from oppertune.core.values import Categorical


class TestExponentialWeights:
    def test_exp_weights_single_param(self) -> None:
        """Test working of exponential weights with a single categorical parameters."""
        random_seed = 12345
        rng = Random(random_seed)

        def calculate_reward(prediction: Mapping[str, str]) -> float:
            return rng.uniform(0, 1)

        parameters = [Categorical("p1", val="0", categories=("0", "1", "2"))]
        tuning_instance = ExponentialWeights(parameters, random_seed=random_seed)
        num_iterations = 5

        expected_attributes = [
            {
                "p": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                "p_hat": [0.0, 0.0, 0.0],
                "eta": 0.6051479953058617,
                "delta": 1.0,
            },
            {
                "p": [0.2421022711651218, 0.2421022711651218, 0.5157954576697562],
                "p_hat": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                "eta": 0.42790425110221986,
                "delta": 1.0,
            },
            {
                "p": [0.24450564437135575, 0.24133453946525096, 0.514159816163393],
                "p_hat": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                "eta": 0.34938235798940165,
                "delta": 1.0,
            },
            {
                "p": [0.4345781919539616, 0.1806179101561332, 0.38480389788990504],
                "p_hat": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                "eta": 0.30257399765293086,
                "delta": 0.8660254037844386,
            },
            {
                "p": [0.3890387643928708, 0.16169096815104012, 0.44927026745608895],
                "p_hat": [0.3468975723859336, 0.31287334617727935, 0.340229081436787],
                "eta": 0.27063041079032607,
                "delta": 0.7745966692414834,
            },
        ]
        expected_rewards = [
            0.41661987254534116,
            0.010169169457068361,
            0.8252065092537432,
            0.2986398551995928,
            0.3684116894884757,
        ]

        for i in range(num_iterations):
            exp = expected_attributes[i]
            assert allclose(tuning_instance.p, exp["p"])
            assert allclose(tuning_instance.p_hat, exp["p_hat"])
            assert allclose(tuning_instance.eta, exp["eta"])
            assert allclose(tuning_instance.delta, exp["delta"])

            prediction, _ = tuning_instance.predict()
            reward = calculate_reward(prediction)
            tuning_instance.set_reward(reward)
            assert allclose(reward, expected_rewards[i])

    def test_exp_weights_multiple_params(self) -> None:
        """Test working of exponential weights with two categorical parameters."""
        random_seed = 12345
        rng = Random(random_seed)

        def calculate_reward(prediction: Mapping[str, str]) -> float:
            return rng.uniform(0, 1)

        parameters = [
            Categorical("p1", val="0", categories=("0", "1", "2")),
            Categorical("p2", val="3", categories=("3", "4")),
        ]

        tuning_instance = ExponentialWeights(parameters, random_seed=random_seed)
        num_iterations = 5

        expected_attributes = [
            {
                "p": [
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                ],
                "p_hat": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "eta": 0.5464673624331794,
                "delta": 1.0,
            },
            {
                "p": [
                    0.11211140092306995,
                    0.11211140092306995,
                    0.4394429953846501,
                    0.11211140092306995,
                    0.11211140092306995,
                    0.11211140092306995,
                ],
                "p_hat": [
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                ],
                "eta": 0.3864107776736279,
                "delta": 1.0,
            },
            {
                "p": [
                    0.11181234307590064,
                    0.11181234307590064,
                    0.4382707785086559,
                    0.11181234307590064,
                    0.11181234307590064,
                    0.11447984918774125,
                ],
                "p_hat": [
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                ],
                "eta": 0.3155030788041409,
                "delta": 1.0,
            },
            {
                "p": [
                    0.07866254586250664,
                    0.37513962792214306,
                    0.3083335369444044,
                    0.07866254586250664,
                    0.07866254586250664,
                    0.08053919754593249,
                ],
                "p_hat": [
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                ],
                "eta": 0.2732336812165897,
                "delta": 1.0,
            },
            {
                "p": [
                    0.06359363105308963,
                    0.4948404030671519,
                    0.24926792000870807,
                    0.06359363105308963,
                    0.06359363105308963,
                    0.06511078376487119,
                ],
                "p_hat": [
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                    0.16666666666666666,
                ],
                "eta": 0.2443876339771208,
                "delta": 1.0,
            },
        ]
        expected_rewards = [
            0.41661987254534116,
            0.010169169457068361,
            0.8252065092537432,
            0.2986398551995928,
            0.3684116894884757,
        ]

        for i in range(num_iterations):
            exp = expected_attributes[i]
            assert allclose(tuning_instance.p, exp["p"])
            assert allclose(tuning_instance.p_hat, exp["p_hat"])
            assert allclose(tuning_instance.eta, exp["eta"])
            assert allclose(tuning_instance.delta, exp["delta"])

            prediction, _ = tuning_instance.predict()
            reward = calculate_reward(prediction)
            tuning_instance.set_reward(reward)
            assert allclose(reward, expected_rewards[i])
