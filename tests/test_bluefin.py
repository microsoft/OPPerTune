import unittest
from functools import partial

import numpy as np

from oppertune import ContinuousValue, DiscreteValue, OPPerTune


class TestBluefin(unittest.TestCase):
    def test_getters(self):
        """Test for getters."""
        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=5.0,
                lb=1.0,
                ub=10.0,
            ),
        )

        tuner = OPPerTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback=1,
                eta=1,
                delta=1,
                random_seed=2,
                eta_decay_rate=0.1,
                normalize=False,
            ),
        )

        self.assertEqual(tuner.backend.feedback, 1)
        self.assertEqual(tuner.backend.eta, 1)
        self.assertEqual(tuner.backend.delta, 1)
        self.assertEqual(tuner.backend.eta_decay_rate, 0.1)
        self.assertEqual(tuner.backend.normalize, False)

    def test_duplicate_params(self):
        """Test for same parameter name."""

        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=0.9,
                lb=0.0,
                ub=1.0,
            ),
            ContinuousValue(
                name="p1",
                initial_value=0.5,
                lb=0.0,
                ub=1.0,
            ),
        )

        self.assertRaises(AssertionError, OPPerTune, parameters)

    def test_features_to_predict(self):
        """Bluefin ignores any additional arguments passed to it during predict."""
        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=4.0,
                lb=1.0,
                ub=10.0,
            ),
            ContinuousValue(
                name="p2",
                initial_value=4.0,
                lb=1.0,
                ub=10.0,
            ),
        )

        tuner = OPPerTune(parameters=parameters, algorithm="bluefin")

        self.assertRaises(
            TypeError,
            tuner.predict,
            features=[1, 0, 0],
            tags={"job_type": 1, "data_type": 2},
        )

    def test_consecutive_predict(self):
        """Consecutive predict calls should return the same values."""
        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=4.0,
                lb=1.0,
                ub=10.0,
            ),
            ContinuousValue(
                name="p2",
                initial_value=4.0,
                lb=1.0,
                ub=10.0,
            ),
        )

        tuner = OPPerTune(parameters=parameters, algorithm="bluefin")

        pred1, _ = tuner.predict()
        pred2, _ = tuner.predict()

        self.assertDictEqual(pred1, pred2)

    def test_constraint(self):
        """Test assertions in Constraint."""

        # Initial value < lb
        self.assertRaises(
            AssertionError,
            ContinuousValue,
            "p1",
            0.0,
            lb=1.0,
            ub=10.0,
        )

        # Initial value > ub
        self.assertRaises(
            AssertionError,
            ContinuousValue,
            name="p2",
            initial_value=11.0,
            lb=1.0,
            ub=10.0,
        )

        # lb > ub
        self.assertRaises(
            AssertionError,
            ContinuousValue,
            name="constraint",
            initial_value=5.0,
            lb=10.0,
            ub=1.0,
        )

        # lb not reachable with initial value
        self.assertRaises(
            AssertionError,
            DiscreteValue,
            name="p4",
            initial_value=1,
            lb=0,
            ub=9,
            step_size=2,
        )

        # ub not reachable with initial value
        self.assertRaises(
            AssertionError,
            DiscreteValue,
            name="p5",
            initial_value=100,
            lb=0,
            ub=750,
            step_size=100,
        )

    def test_onepoint(self):
        """Testing onepoint."""

        def get_reward(prediction: np.ndarray):
            """Squared Loss."""
            target = np.array([1, 8, 7, 2])
            return -np.square(prediction - target).sum() / 100

        _EXPECTED_REWARDS = [
            -0.8470379599421056,
            -1.090085203321628,
            -0.7727964449646174,
            -0.9042699559720375,
            -0.7267558202557508,
            -0.7735871899959318,
            -0.41670744890642736,
            -0.7807568127908868,
            -0.9660255197842609,
            -0.6239352884222221,
            -0.5096711906872058,
            -0.8867704637651685,
            -0.5369274960850529,
            -0.8632963898554757,
            -0.7071389714407826,
            -0.6807698472816998,
            -0.5263770684402961,
            -0.587805573352956,
            -0.5280901701861775,
            -0.7500028270255834,
            -0.9293010571833566,
            -1.130279769406053,
            -1.0812610293565248,
            -1.0452795990064792,
            -0.6580076407398049,
            -0.6988750493391594,
            -0.9318474021143408,
            -1.0266518904981554,
            -0.795425007853701,
            -0.8238599014648174,
            -1.0448787587659825,
            -1.0637833794130862,
            -0.5925626381944262,
            -0.8292773854139002,
            -0.5863878533592942,
            -0.6758741945761417,
            -0.5780348742007179,
            -0.5390637968216082,
            -0.37694070573131566,
            -0.7028940623976796,
            -0.5701598167499773,
            -1.2474885383057925,
            -0.4004231110053077,
            -0.24741703912017418,
            -0.1317689035782558,
            -0.23699370130629208,
            -0.13321801546061748,
            -0.20582465478704773,
            -0.1445163092100765,
            -0.11792887548476633,
        ]

        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=5.0,
                lb=1.0,
                ub=10.0,
            ),
            ContinuousValue(
                name="p2",
                initial_value=3.0,
                lb=1.0,
                ub=10.0,
            ),
            DiscreteValue(
                name="p3",
                initial_value=2,
                lb=1,
                ub=10,
            ),
            DiscreteValue(
                name="p4",
                initial_value=7,
                lb=1,
                ub=10,
            ),
        )

        tuner = OPPerTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                normalize=False,
                feedback=1,
                eta=1,
                delta=1,
                random_seed=2,
            ),
        )
        num_iterations = 50

        rewards = []
        for _ in range(num_iterations):
            pred, _ = tuner.predict()

            reward = get_reward(np.asarray(list(pred.values())))
            rewards.append(reward)

            tuner.set_reward(reward)

        for r, exp_r in zip(rewards, _EXPECTED_REWARDS):
            self.assertAlmostEqual(r, exp_r, places=9)

    def test_onepoint_normalized(self):
        """Testing normalization with onepoint feedback."""

        def get_reward(prediction: np.ndarray):
            """Squared Loss."""
            target = np.array([10, 50, 70, 20, 80])
            return -np.square(prediction - target).sum() / (100**2)

        _EXPECTED_REWARDS = [
            -1.0090941689799096,
            -1.3842386961672057,
            -0.8303832549312993,
            -1.2270727207429466,
            -0.5891492029240215,
            -0.41051854998703874,
            -0.6623820066628344,
            -0.4614498034177098,
            -0.7835490811529603,
            -0.6988268656309505,
            -1.0729558972338582,
            -0.505733368524754,
            -0.6120121173225832,
            -0.6913677653662751,
            -0.32064776509846804,
            -0.15225518542962946,
            -0.1847000810856561,
            -0.27090069125385735,
            -0.18144987939461998,
            -0.23758935073922677,
            -0.2321903407908043,
            -0.2881257812445579,
            -0.11983896605782596,
            -0.1472093558637525,
            -0.26827233184800564,
            -0.22104473207304753,
            -0.2139390187705355,
            -0.37680166837666595,
            -0.2540939495146872,
            -0.1890471538854583,
            -0.2909817802747426,
            -0.26012397242082386,
            -0.30619619075695675,
            -0.40259590302066633,
            -0.527427214620143,
            -0.8236439782237529,
            -1.249469266217643,
            -0.6818068981335881,
            -0.3043094812006053,
            -0.3798475306799169,
            -0.1534239038179127,
            -0.2447383410966443,
            -0.355578754136262,
            -0.20967475104756594,
            -0.1660062226864054,
            -0.2207246168143225,
            -0.230656104156308,
            -0.0710706084705719,
            -0.13721431126548797,
            -0.09564408759764734,
        ]

        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=50.0,
                lb=1.0,
                ub=100.0,
            ),
            ContinuousValue(
                name="p2",
                initial_value=20.0,
                lb=1.0,
                ub=100.0,
            ),
            ContinuousValue(
                name="p3",
                initial_value=30.0,
                lb=1.0,
                ub=100.0,
            ),
            DiscreteValue(
                name="p4",
                initial_value=70,
                lb=1,
                ub=100,
            ),
            DiscreteValue(
                name="p5",
                initial_value=10,
                lb=1,
                ub=100,
            ),
        )

        tuner = OPPerTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback=1,
                eta=0.01,
                delta=0.1,
                random_seed=2,
                normalize=True,
            ),
        )
        num_iterations = 50

        rewards = []
        for _ in range(num_iterations):
            pred, _ = tuner.predict()

            reward = get_reward(np.asarray(list(pred.values())))
            rewards.append(reward)

            tuner.set_reward(reward)

        for r, exp_r in zip(rewards, _EXPECTED_REWARDS):
            self.assertAlmostEqual(r, exp_r, places=9)

    def test_twopoint_normalized(self):
        """Testing normalization with twopoint feedback."""

        def get_reward(prediction: np.ndarray):
            """Absolute Loss."""
            target = np.array([200, 800])
            return -np.abs(prediction - target).sum() / 1000

        _EXPECTED_REWARDS = [
            -0.6639602790056434,
            -0.5360397209943567,
            -0.6285948688971376,
            -0.5467595767235773,
            -0.5975571799665299,
            -0.5672996218985927,
            -0.5181436902419331,
            -0.6452028438283908,
            -0.6336557395873713,
            -0.5050158987274351,
            -0.6157274505999942,
            -0.4971808662269632,
            -0.5703539283791988,
            -0.5200301733522494,
            -0.5977233518998105,
            -0.48827736542341077,
            -0.5972310457511533,
            -0.4689795812920483,
            -0.5702248173042045,
            -0.4687798448757037,
            -0.5441762554113116,
            -0.4769954266964745,
            -0.49984307724932336,
            -0.5131311070336015,
            -0.4455513725058679,
            -0.5670826083390285,
            -0.5536276870148533,
            -0.4332498519999784,
            -0.47634314403355005,
            -0.48464408146319027,
            -0.5155202008093503,
            -0.44532635691986067,
            -0.4870685375043506,
            -0.46431973222001166,
            -0.4666371710425773,
            -0.48369942682216965,
            -0.42717851395657874,
            -0.5225622301160502,
            -0.4426235278500334,
            -0.4898771641235309,
            -0.4298142473847923,
            -0.49817210934764644,
            -0.46874165889692765,
            -0.44995316497942767,
            -0.4054896910500516,
            -0.5124589992677925,
            -0.40296537123613757,
            -0.4928575177570906,
            -0.3870450312034872,
            -0.49252862695322025,
        ]

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
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback=2,
                eta=0.0025,
                delta=0.05,
                optimizer="sgd",
                normalize=True,
                eta_decay_rate=0.03,
                random_seed=2,
            ),
        )
        num_iterations = 50

        rewards = []
        for _ in range(num_iterations):
            pred, _ = tuner.predict()

            reward = get_reward(np.asarray(list(pred.values())))
            rewards.append(reward)

            tuner.set_reward(reward)

        for r, exp_r in zip(rewards, _EXPECTED_REWARDS):
            self.assertAlmostEqual(r, exp_r, places=9)

    def test_twopoint(self):
        """Testing twopoint."""

        def get_reward(prediction: np.ndarray):
            """Absolute Loss."""
            target = np.array([2, 8])
            return -np.abs(prediction - target).sum() / 10

        _EXPECTED_REWARDS = [
            -0.7,
            -0.5,
            -0.7,
            -0.5,
            -0.6,
            -0.5,
            -0.4,
            -0.6,
            -0.6,
            -0.4,
            -0.5,
            -0.4,
            -0.5,
            -0.4,
            -0.5,
            -0.4,
            -0.5,
            -0.3,
            -0.4,
            -0.3,
            -0.4,
            -0.2,
            -0.3,
            -0.3,
            -0.2,
            -0.4,
            -0.4,
            -0.2,
            -0.3,
            -0.2,
            -0.3,
            -0.3,
            -0.2,
            -0.3,
            -0.2,
            -0.2,
            -0.1,
            -0.3,
            -0.2,
            -0.2,
            0.0,
            -0.2,
            -0.1,
            -0.2,
            0.0,
            -0.2,
            -0.1,
            0.0,
            -0.1,
            -0.2,
        ]

        parameters = (
            DiscreteValue(
                name="p1",
                initial_value=4,
                lb=1,
                ub=10,
            ),
            DiscreteValue(
                name="p2",
                initial_value=4,
                lb=1,
                ub=10,
            ),
        )

        tuner = OPPerTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback=2,
                eta=0.01,
                delta=0.1,
                optimizer="sgd",
                random_seed=2,
            ),
        )
        num_iterations = 50

        rewards = []
        for _ in range(num_iterations):
            pred, _ = tuner.predict()

            reward = get_reward(np.asarray(list(pred.values())))
            rewards.append(reward)

            tuner.set_reward(reward)

        self.assertListEqual(rewards, _EXPECTED_REWARDS)

    def test_twopoint_zero_reward(self):
        """Test if zero division error occurs when reward is zero."""

        def get_reward():
            return 0

        parameters = (
            DiscreteValue(
                name="p1",
                initial_value=4,
                lb=1,
                ub=10,
            ),
            DiscreteValue(
                name="p2",
                initial_value=4,
                lb=1,
                ub=10,
            ),
        )

        tuner = OPPerTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback=2,
                eta=0.01,
                delta=0.1,
                optimizer="sgd",
                random_seed=2,
            ),
        )
        num_iterations = 5

        for _ in range(num_iterations):
            tuner.predict()

            reward = get_reward()

            tuner.set_reward(reward)

    def test_step_size(self):
        """Testing Step size."""

        def get_reward(prediction: np.ndarray):
            """Squared Loss."""
            target = np.array([1, 700])
            return -np.square(prediction - target).sum() / (1000**2)

        _EXPECTED_REWARDS = [
            -0.250009,
            -0.250025,
            -0.250025,
            -0.250009,
            -0.250009,
        ]

        parameters = (
            DiscreteValue(
                name="p1",
                initial_value=5,
                lb=0,
                ub=10,
            ),
            ContinuousValue(
                name="p2",
                initial_value=100.0,
                lb=100.0,
                ub=900.0,
                step_size=100.0,
            ),
        )

        tuner = OPPerTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback=2,
                eta=0.01,
                delta=0.1,
                optimizer="sgd",
                random_seed=4,
            ),
        )
        num_iterations = 5

        rewards = []
        for _ in range(num_iterations):
            pred, _ = tuner.predict()

            reward = get_reward(np.asarray(list(pred.values())))
            rewards.append(reward)

            tuner.set_reward(reward)

        self.assertListEqual(rewards, _EXPECTED_REWARDS)

    def test_zero_eta(self):
        """Zero eta should return the same parameters."""
        parameters = (
            DiscreteValue(
                name="p1",
                initial_value=5,
                lb=0,
                ub=10,
            ),
            ContinuousValue(
                name="p2",
                initial_value=100.0,
                lb=100.0,
                ub=900.0,
                step_size=100.0,
            ),
        )

        tuner = OPPerTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback=2,
                eta=0.0,
                delta=1e-8,
                optimizer="sgd",
                random_seed=4,
            ),
        )
        num_iterations = 5

        for _ in range(num_iterations):
            pred, _ = tuner.predict()

            for p in parameters:
                self.assertEqual(p.initial_value, pred[p.name])

            reward = -1  # Arbitrary reward

            tuner.set_reward(reward)

        parameters = (
            DiscreteValue(
                name="p1",
                initial_value=64,
                lb=64,
                ub=19064,
                step_size=1000,
            ),
            DiscreteValue(
                name="p2",
                initial_value=40,
                lb=10,
                ub=40,
                step_size=10,
            ),
            DiscreteValue(
                name="p3",
                initial_value=64,
                lb=64,
                ub=100,
                step_size=1,
            ),
            DiscreteValue(
                name="p4",
                initial_value=40,
                lb=1,
                ub=40,
                step_size=1,
            ),
            ContinuousValue(
                name="p5",
                initial_value=64.0,
                lb=64.0,
                ub=19064.0,
                step_size=1000.0,
            ),
            ContinuousValue(
                name="p6",
                initial_value=40.0,
                lb=10.0,
                ub=40.0,
                step_size=10.0,
            ),
            ContinuousValue(
                name="p7",
                initial_value=64.0,
                lb=64.0,
                ub=19064.0,
                step_size=1.0,
            ),
            ContinuousValue(
                name="p8",
                initial_value=40.0,
                lb=1.0,
                ub=40.0,
                step_size=1.0,
            ),
        )

        tuner = OPPerTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback=2,
                eta=0.0,
                delta=0.1,
                optimizer="sgd",
                random_seed=4,
            ),
        )
        num_iterations = 5

        param_values = []
        for _ in range(num_iterations):
            pred, _ = tuner.predict()

            for p in parameters:
                self.assertEqual(p.initial_value, pred[p.name])

            reward = -1  # Arbitrary reward
            param_values.append(pred)

            tuner.set_reward(reward)

    def test_zero_delta(self):
        """Zero delta should raise an error."""

        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=0.9,
                lb=0.0,
                ub=1.0,
            ),
            ContinuousValue(
                name="p1",
                initial_value=0.5,
                lb=0.0,
                ub=1.0,
            ),
        )

        tuner_func = partial(
            OPPerTune,
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback=2,
                eta=0.0,
                delta=0.0,
                optimizer="sgd",
                random_seed=4,
            ),
        )

        self.assertRaises(AssertionError, tuner_func)

    def test_rmsprop(self):
        """Testing the RMSProp optimizer."""

        def get_reward(pred: np.ndarray):
            """Root Mean Squared Error."""
            target = np.asarray([0.1, 0.3, 0.6])
            return -np.sqrt(np.mean(np.square(pred - target)))

        _EXPECTED_PARAM_VALUES = [
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

        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=0.87362385,
                lb=0.0,
                ub=1.0,
            ),
            ContinuousValue(
                name="p2",
                initial_value=0.30238590,
                lb=0.0,
                ub=1.0,
            ),
            ContinuousValue(
                name="p3",
                initial_value=0.10718888,
                lb=0.0,
                ub=1.0,
            ),
        )

        tuner = OPPerTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback=2,
                eta=0.01,
                delta=0.1,
                optimizer="rmsprop",
                random_seed=4,
            ),
        )

        num_iterations = 10

        param_values = []
        for _ in range(num_iterations):
            pred, _ = tuner.predict()

            reward = get_reward(np.asarray(list(pred.values())))
            param_values.append(pred)

            tuner.set_reward(reward)

        for idx, values in enumerate(param_values):
            for param in parameters:
                self.assertAlmostEqual(values[param.name], _EXPECTED_PARAM_VALUES[idx][param.name])
