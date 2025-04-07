from random import Random
from typing import Mapping

from numpy import allclose

from oppertune.algorithms.exponential_weights_slates import ExponentialWeightsSlates
from oppertune.core.values import Categorical


class TestExponentialWeightsSlates:
    def test_exp_weights_single_param(self) -> None:
        """Test working of exponential weights with a single categorical parameters."""
        random_seed = 12345
        rng = Random(random_seed)

        def calculate_reward(prediction: Mapping[str, str]) -> float:
            return rng.uniform(0, 1)

        parameters = [Categorical("p1", val="0", categories=("0", "1", "2"))]
        tuning_instance = ExponentialWeightsSlates(parameters, random_seed=random_seed)
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
            # for j in range(len(tuning_instance.p[0])):
            # p[-1].append(tuning_instance.p[0][j])
            # p_hat[-1].append(tuning_instance.p_hat[0][j])
            # eta.append(tuning_instance.eta[0])
            # delta.append(tuning_instance.delta[0])

            assert allclose(tuning_instance.p, exp["p"])
            assert allclose(tuning_instance.p_hat, exp["p_hat"])
            assert allclose(tuning_instance.eta, exp["eta"])
            assert allclose(tuning_instance.delta, exp["delta"])

            prediction, _ = tuning_instance.predict()
            reward = calculate_reward(prediction)
            tuning_instance.set_reward(reward)
            assert allclose(reward, expected_rewards[i])

        # print(f"\np = {p}")
        # print(f"p_hat = {p_hat}")
        # print(f"eta = {eta}")
        # print(f"delta = {delta}")

    def test_exp_weights_multiple_params(self) -> None:
        """Test working of exponential weights with two categorical parameters."""
        random_seed = 12345
        rng = Random(random_seed)

        def calculate_reward(prediction: Mapping[str, str]) -> float:
            return rng.uniform(0, 1)

        parameters = [
            Categorical("p1", val="0", categories=("0", "1", "2")),
            Categorical("p2", val="3", categories=("3", "4", "5")),
        ]

        tuning_instance = ExponentialWeightsSlates(parameters, random_seed=random_seed)
        num_iterations = 5

        expected_attributes = [
            {
                "p": [
                    [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                    [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                ],
                "p_hat": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "eta": [0.6051479953058617, 0.6051479953058617],
                "delta": [1.0, 1.0],
            },
            {
                "p": [
                    [0.2421022711651218, 0.2421022711651218, 0.5157954576697562],
                    [0.5157954576697562, 0.2421022711651218, 0.2421022711651218],
                ],
                "p_hat": [
                    [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                    [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                ],
                "eta": [0.42790425110221986, 0.42790425110221986],
                "delta": [1.0, 1.0],
            },
            {
                "p": [
                    [0.24450564437135575, 0.24133453946525096, 0.514159816163393],
                    [0.514159816163393, 0.24133453946525096, 0.24450564437135575],
                ],
                "p_hat": [
                    [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                    [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                ],
                "eta": [0.34938235798940165, 0.34938235798940165],
                "delta": [1.0, 1.0],
            },
            {
                "p": [
                    [0.14324570099885098, 0.14138788235264435, 0.7153664166485044],
                    [0.7153664166485044, 0.14138788235264435, 0.14324570099885098],
                ],
                "p_hat": [
                    [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                    [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                ],
                "eta": [0.30257399765293086, 0.30257399765293086],
                "delta": [0.8660254037844386, 0.8660254037844386],
            },
            {
                "p": [
                    [0.18315990416833275, 0.134800947605418, 0.6820391482262491],
                    [0.6824221772475965, 0.180928909856866, 0.13664891289553727],
                ],
                "p_hat": [
                    [0.30786641954574895, 0.3076175190427817, 0.3845160614114693],
                    [0.3845160614114693, 0.3076175190427817, 0.30786641954574895],
                ],
                "eta": [0.27063041079032607, 0.27063041079032607],
                "delta": [0.7745966692414834, 0.7745966692414834],
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
