import random
import unittest

import numpy as np

from oppertune import ContinuousValue, DiscreteValue, OPPerTune


class TestDGT(unittest.TestCase):
    def test_onepoint_leaf_weights_fixed(self):
        def get_reward(pred, jobtype):
            return -np.sqrt(np.mean(np.square(pred - _OPTIMAL_PARAMS[jobtype])))

        _OPTIMAL_PARAMS = [
            [0.1, 0.3, 0.6],
            [0.4, 0.5, 0.1],
            [0.2, 0.3, 0.7],
            [0.1, 0.7, 0.8],
            [0.6, 0.7, 0.9],
            [0.7, 0.8, 0.3],
            [0.4, 0.6, 0.2],
            [0.3, 0.2, 0.8],
        ]
        _OPTIMAL_PARAMS = np.asarray(_OPTIMAL_PARAMS)

        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=0.5,
                lb=0.0,
                ub=1.0,
            ),
            ContinuousValue(
                name="p2",
                initial_value=0.3,
                lb=0.0,
                ub=1.0,
            ),
            ContinuousValue(
                name="p3",
                initial_value=0.1,
                lb=0.0,
                ub=1.0,
            ),
        )

        features = (
            {
                "name": "jobtype",
                "values": [0, 1, 2, 3, 4, 5, 6, 7],
            },
        )

        internal_weights = [
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, 1, -1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, -1, -1],
            [1, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, -1],
        ]

        leaf_weights = [
            [0.8736, 0.5378, 0.8781, 0.6719, 0.6582, 0.2729, 0.8810, 0.1050],
            [0.3024, 0.4478, 0.7235, 0.2581, 0.7904, 0.8867, 0.2311, 0.5779],
            [0.1072, 0.4093, 0.1353, 0.8653, 0.4489, 0.8592, 0.7290, 0.7930],
        ]

        tuner = OPPerTune(
            algorithm="dgt",
            parameters=parameters,
            algorithm_args=dict(
                features=features,
                height=3,
                feedback=1,
                eta1=0.01,
                eta2=0.01,
                delta=0.1,
                random_seed=4,
                optimizer="sgd",
                internal_weights=internal_weights,
                fix_internal_weights=True,
                leaf_weights=leaf_weights,
            ),
        )

        _EXPECTED_INTERNAL_WEIGHTS = [
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, 1, -1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, -1, -1],
            [1, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, -1],
        ]

        _EXPECTED_LEAF_WEIGHTS = [
            [0.85067399, 0.55296339, 0.78706909, 0.81520554, 0.68358037, 0.14021085, 0.85562129, 0.10500000],
            [0.28008001, 0.44418510, 0.87799699, 0.16190697, 0.85815041, 0.90000000, 0.19018032, 0.57790000],
            [0.10000000, 0.44841306, 0.10023468, 0.90000000, 0.49938970, 0.81616215, 0.84867145, 0.79300000],
        ]

        jobs = [2, 5, 1, 0, 6, 5, 4, 3, 2, 3]
        for jobtype in jobs:
            cur_features = {"jobtype": jobtype}
            pred, _metadata = tuner.predict(features=cur_features)

            pred_arr = np.asarray(list(pred.values()))
            reward = get_reward(pred_arr, jobtype)

            tuner.set_reward(reward, metadata=_metadata)
            self.assertEqual(jobtype, _metadata["selected_leaf"])

        trained_internal_weights = tuner.backend._predicate_layers[0].weight.detach().numpy()
        trained_leaf_weights = tuner.backend._or_layer.weight.detach().numpy()

        self.assertTrue(np.allclose(_EXPECTED_INTERNAL_WEIGHTS, trained_internal_weights))
        self.assertTrue(np.allclose(_EXPECTED_LEAF_WEIGHTS, trained_leaf_weights))

    def test_twopoint(self):
        def get_reward(pred, jobtype):
            return -np.sqrt(np.mean(np.square(pred - _OPTIMAL_PARAMS[jobtype])))

        _OPTIMAL_PARAMS = [
            [0.1, 0.3, 0.6],
            [0.4, 0.5, 0.1],
            [0.2, 0.3, 0.7],
            [0.1, 0.7, 0.8],
            [0.6, 0.7, 0.9],
            [0.7, 0.8, 0.3],
            [0.4, 0.6, 0.2],
            [0.3, 0.2, 0.8],
        ]
        _OPTIMAL_PARAMS = np.asarray(_OPTIMAL_PARAMS)

        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=0.5,
                lb=0.0,
                ub=1.0,
            ),
            ContinuousValue(
                name="p2",
                initial_value=0.3,
                lb=0.0,
                ub=1.0,
            ),
            ContinuousValue(
                name="p3",
                initial_value=0.1,
                lb=0.0,
                ub=1.0,
            ),
        )

        features = (
            {
                "name": "jobtype",
                "values": [0, 1, 2, 3, 4, 5, 6, 7],
            },
        )

        internal_weights = [
            [0.0422, 0.0418, -0.2889, -0.2051, -0.3485, -0.326, 0.3485, 0.2921],
            [0.0838, 0.3354, -0.128, -0.2016, 0.3014, -0.0187, 0.0671, 0.209],
            [0.1863, -0.2024, -0.1367, -0.3263, 0.0156, -0.1268, 0.0759, 0.0164],
            [0.3014, 0.0305, 0.1772, 0.1116, 0.3137, -0.147, 0.2952, 0.2536],
            [-0.1411, 0.2167, 0.3269, -0.1999, -0.1379, 0.2419, 0.2851, 0.0933],
            [0.1086, 0.2619, 0.3098, -0.2236, -0.1454, 0.2977, 0.191, -0.1659],
            [-0.12, 0.3431, -0.2214, 0.2274, 0.2052, -0.3391, -0.3529, 0.1174],
        ]

        leaf_weights = [
            [0.2776, 0.7966, 0.2654, 0.8349, 0.4907, 0.5894, 0.7127, 0.5147],
            [0.3374, 0.2502, 0.1646, 0.6908, 0.4530, 0.2266, 0.8039, 0.3193],
            [0.4314, 0.3369, 0.6030, 0.5639, 0.5799, 0.3127, 0.3277, 0.3029],
        ]

        tuner = OPPerTune(
            algorithm="dgt",
            parameters=parameters,
            algorithm_args=dict(
                features=features,
                height=3,
                feedback=2,
                eta1=0.01,
                eta2=0.01,
                delta=0.1,
                random_seed=4,
                optimizer="rmsprop",
                internal_weights=internal_weights,
                fix_internal_weights=False,
                leaf_weights=leaf_weights,
            ),
        )

        _EXPECTED_INTERNAL_WEIGHTS = [
            [-0.12598589, -0.05819987, -0.38889860, -0.10510005, -0.24850004, -0.44758348, 0.34850000, 0.29164685],
            [0.13283667, 0.23540016, -0.22799068, -0.30159952, 0.20140014, -0.11038836, 0.06710000, 0.04403482],
            [-0.01249462, -0.30239907, -0.23669930, -0.22630002, 0.11559995, -0.24138136, 0.07590000, 0.12549839],
            [0.49924224, 0.13049993, 0.27719709, 0.21159988, 0.41369993, -0.01451198, 0.29520000, 0.25367533],
            [-0.11643034, 0.31669953, 0.22690030, -0.09990004, -0.03790012, 0.11852667, 0.28510000, 0.29329500],
            [0.10591219, 0.36189977, 0.40979948, -0.12360015, -0.04540005, 0.43633881, 0.19100000, -0.13184488],
            [-0.30994403, 0.44309283, -0.12140031, 0.12740003, 0.10520012, -0.23122056, -0.35290000, 0.11820633],
        ]

        _EXPECTED_LEAF_WEIGHTS = [
            [0.30247984, 0.79660000, 0.36540000, 0.73490003, 0.39070001, 0.48940001, 0.61270002, 0.37091282],
            [0.49256160, 0.25020000, 0.10000000, 0.79079986, 0.55299999, 0.12660004, 0.70390001, 0.45431196],
            [0.52854119, 0.33690000, 0.70300000, 0.46390001, 0.47990001, 0.41270000, 0.42770000, 0.50363515],
        ]

        jobs = [4, 4, 5, 5, 0, 0, 7, 7, 3, 3, 0, 0, 2, 2, 1, 1, 5, 5, 7, 7]
        for jobtype in jobs:
            cur_features = {"jobtype": jobtype}

            pred, _metadata = tuner.predict(features=cur_features)

            pred_arr = np.asarray(list(pred.values()))
            reward = get_reward(pred_arr, jobtype)

            tuner.set_reward(reward, metadata=_metadata)

        trained_internal_weights = tuner.backend._predicate_layers[0].weight.detach().numpy()
        trained_leaf_weights = tuner.backend._or_layer.weight.detach().numpy()

        self.assertTrue(np.allclose(_EXPECTED_INTERNAL_WEIGHTS, trained_internal_weights))
        self.assertTrue(np.allclose(_EXPECTED_LEAF_WEIGHTS, trained_leaf_weights))

    def test_dgt_with_fixed_weights(self):
        """Each leaf in DGT is like a bluefin instance."""

        def get_reward(pred: np.ndarray):
            _OPTIMAL_PARAM = np.asarray([100, 500, 900])
            return -np.sqrt(np.mean(np.square(pred - _OPTIMAL_PARAM)))

        random_seed = 4

        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=500,
                lb=0.0,
                ub=1000.0,
            ),
            ContinuousValue(
                name="p2",
                initial_value=300,
                lb=0.0,
                ub=1000.0,
            ),
            ContinuousValue(
                name="p3",
                initial_value=100,
                lb=0.0,
                ub=1000.0,
            ),
        )

        features = [
            {
                "name": "jobtype",
                "values": (0, 1, 2, 3),
            },
        ]

        internal_weights = [
            [0.1, 0.1, -0.1, -0.1],
            [0.1, -0.1, 0, 0],
            [0, 0, 0.1, -0.1],
        ]

        tuner_dgt = OPPerTune(
            algorithm="dgt",
            parameters=parameters,
            algorithm_args=dict(
                features=features,
                height=2,
                feedback=2,
                eta1=0.01,
                eta2=0.01,
                delta=0.1,
                random_seed=random_seed,
                optimizer="rmsprop",
                internal_weights=internal_weights,
                fix_internal_weights=True,
                leaf_weights=None,
            ),
        )

        tuner_bluefin = OPPerTune(
            parameters=parameters,
            algorithm="bluefin",
            algorithm_args=dict(
                feedback=2,
                eta=0.01,
                delta=0.1,
                optimizer="rmsprop",
                random_seed=random_seed,
                normalize=True,
            ),
        )

        jobtype = 0
        num_iterations = 100
        for _ in range(num_iterations):
            # DGT
            cur_features = {"jobtype": jobtype}
            pred, _metadata = tuner_dgt.predict(features=cur_features)
            pred_dgt = np.asarray(list(pred.values()))
            reward = get_reward(pred_dgt)
            tuner_dgt.set_reward(reward, metadata=_metadata)
            self.assertEqual(jobtype, _metadata["selected_leaf"])

            # Bluefin
            pred, _metadata = tuner_bluefin.predict()
            pred_bf = np.asarray(list(pred.values()))
            reward = get_reward(pred_bf)
            tuner_bluefin.set_reward(reward, metadata=_metadata)

            # Check if the predictions are the same
            np.testing.assert_almost_equal(pred_dgt, pred_bf, decimal=4)

        # Center for jobtype=0
        dgt_center = [
            tuner_dgt.backend._or_layer.weight.data[0][0].item(),
            tuner_dgt.backend._or_layer.weight.data[1][0].item(),
            tuner_dgt.backend._or_layer.weight.data[2][0].item(),
        ]
        bluefin_center = tuner_bluefin.backend.w_center

        # Check if the centres are same
        for idx, val in enumerate(bluefin_center):
            self.assertAlmostEqual(dgt_center[idx], val, 4)

    def test_dgt_features(self):
        parameters = (
            DiscreteValue(
                name="p1",
                initial_value=1000,
                lb=0,
                ub=10000,
            ),
            DiscreteValue(
                name="p2",
                initial_value=9000,
                lb=0,
                ub=10000,
            ),
        )

        features = (
            {"name": "n1", "values": ["small", "medium", "big"]},
            {"name": "n2", "values": ["0", "1"]},
        )

        internal_weights = [
            [1, -1, -1, 0, 0],  # small left, (medium, large) right
            [1, 0, 0, 0, 0],  # small left
            [0, 1, -1, 0, 0],  # medium left, large right
            [0, 0, 0, 1, -1],  # 0 left, 1 right
            [0, 0, 0, 0, 0],  # Node never visited
            [0, 0, 0, 1, -1],  # 0 left, 1 right
            [0, 0, 0, 1, -1],  # 0 left, 1 right
        ]

        leaf_weights = [
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
            [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        ]

        tuner_dgt = OPPerTune(
            algorithm="dgt",
            parameters=parameters,
            algorithm_args=dict(
                features=features,
                height=3,
                feedback=2,
                eta1=1e-16,
                eta2=1e-16,
                delta=1e-8,  # A very low value so that no perturbation happens
                random_seed=4,
                optimizer="rmsprop",
                internal_weights=internal_weights,
                fix_internal_weights=True,
                leaf_weights=leaf_weights,
            ),
        )

        n1_values = ["small", "small", "medium", "medium", "big", "big"]
        n2_values = ["0", "1", "0", "1", "0", "1"]

        _EXPECTED_PREDS = [
            {"p1": 500, "p2": 4500},
            {"p1": 1000, "p2": 5000},
            {"p1": 2500, "p2": 6500},
            {"p1": 3000, "p2": 7000},
            {"p1": 3500, "p2": 7500},
            {"p1": 4000, "p2": 8000},
        ]

        for i in range(6):
            features = {"n1": n1_values[i], "n2": n2_values[i]}
            pred, _ = tuner_dgt.predict(features=features)
            expected_pred = _EXPECTED_PREDS[i]
            for key in pred.keys():
                self.assertEqual(pred[key], expected_pred[key])

    def test_dgt_eta1_eta2(self):
        def get_reward(pred: np.ndarray, jobtype):
            _OPTIMAL_PARAM = np.asarray([[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]])
            return -np.sqrt(np.mean(np.square(pred - _OPTIMAL_PARAM[jobtype])))

        parameters = (
            DiscreteValue(
                name="p1",
                initial_value=100,
                lb=0,
                ub=1000,
            ),
            DiscreteValue(
                name="p2",
                initial_value=100,
                lb=0,
                ub=1000,
            ),
            DiscreteValue(
                name="p3",
                initial_value=100,
                lb=0,
                ub=1000,
            ),
        )

        features = (
            {
                "name": "jobtype",
                "values": [0, 1, 2, 3],
            },
        )

        internal_weights = [
            [1, 1, -1, -1],
            [1, -1, 0, 0],
            [0, 0, 1, -1],
        ]

        leaf_weights = [
            [0.015, 0.045, 0.075, 0.105],
            [0.025, 0.055, 0.085, 0.115],
            [0.035, 0.065, 0.095, 0.125],
        ]

        tuner_dgt = OPPerTune(
            algorithm="dgt",
            parameters=parameters,
            algorithm_args=dict(
                features=features,
                height=2,
                feedback=2,
                eta1=1e-16,  # Internal nodes should not change much
                eta2=0.01,
                delta=0.1,
                random_seed=4,
                optimizer="rmsprop",
                internal_weights=internal_weights,
                fix_internal_weights=False,
                leaf_weights=leaf_weights,
            ),
        )

        num_iterations = 1000
        jobtypes_list = [0, 1, 2, 3]
        for _ in range(num_iterations):
            jobtype = random.choice(jobtypes_list)
            features = {"jobtype": jobtype}

            pred, _metadata = tuner_dgt.predict(features=features)
            reward = get_reward(np.asarray(list(pred.values())), jobtype=jobtype)
            tuner_dgt.set_reward(reward, metadata=_metadata)

            pred, _metadata = tuner_dgt.predict(features=features)
            reward = get_reward(np.asarray(list(pred.values())), jobtype=jobtype)
            tuner_dgt.set_reward(reward, metadata=_metadata)

        # The values should have converged
        for jobtype in jobtypes_list:
            features = {"jobtype": jobtype}

            pred, _metadata = tuner_dgt.predict(features=features)
            reward = get_reward(np.asarray(list(pred.values())), jobtype=jobtype)

            self.assertLessEqual(reward, -0.15)

        learned_internal_weights = tuner_dgt.backend._predicate_layers[0].weight.data

        for i in range(len(internal_weights)):
            for j in range(len(internal_weights[i])):
                self.assertAlmostEqual(internal_weights[i][j], learned_internal_weights[i][j].item())

    def test_dgt_feature_map_selected_leaf(self):
        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=0.9,
                lb=0.0,
                ub=1.0,
            ),
            ContinuousValue(
                name="p2",
                initial_value=0.2,
                lb=0.0,
                ub=1.0,
            ),
            ContinuousValue(
                name="p3",
                initial_value=0.6,
                lb=0.0,
                ub=1.0,
            ),
        )

        features = [
            {"name": "f1", "values": ("false", "true")},
            {"name": "f2", "values": ("false", "true")},
            {"name": "f3", "values": ("false", "true")},
        ]

        internal_weights = [
            [0.1, -0.1, 0, 0, 0, 0],
            [0, 0, 0.1, -0.1, 0, 0],
            [0, 0, 0.1, -0.1, 0, 0],
            [0, 0, 0, 0, 0.1, -0.1],
            [0, 0, 0, 0, 0.1, -0.1],
            [0, 0, 0, 0, 0.1, -0.1],
            [0, 0, 0, 0, 0.1, -0.1],
        ]

        leaf_weights = [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
        ]

        tuner_dgt = OPPerTune(
            algorithm="dgt",
            parameters=parameters,
            algorithm_args=dict(
                features=features,
                height=3,
                feedback=2,
                eta1=0.01,
                eta2=0.01,
                delta=1e-16,  # Limit perturbations
                random_seed=4,
                optimizer="rmsprop",
                internal_weights=internal_weights,
                fix_internal_weights=True,
                leaf_weights=leaf_weights,
            ),
        )

        f1_features = ["false", "false", "false", "false", "true", "true", "true", "true"]
        f2_features = ["false", "false", "true", "true", "false", "false", "true", "true"]
        f3_features = ["false", "true", "false", "true", "false", "true", "false", "true"]

        for i in range(8):
            features = {"f1": f1_features[i], "f2": f2_features[i], "f3": f3_features[i]}

            pred, _metadata = tuner_dgt.predict(features=features)
            self.assertEqual(_metadata["selected_leaf"], i)
            self.assertAlmostEqual(pred["p1"], leaf_weights[0][i])
            self.assertAlmostEqual(pred["p2"], leaf_weights[1][i])
            self.assertAlmostEqual(pred["p3"], leaf_weights[2][i])

    def test_convergence_with_non_sequential_predicts(self):
        """Check if DGT converges with non sequential predicts."""
        rng = np.random.default_rng(4)

        def get_reward(pred, jobtype):
            return -np.sqrt(np.mean(np.square(pred - _OPTIMAL_PARAMS[jobtype])))

        _OPTIMAL_PARAMS = [
            [0.1, 0.3, 0.6],
            [0.4, 0.5, 0.1],
            [0.2, 0.3, 0.7],
            [0.1, 0.7, 0.8],
            [0.6, 0.7, 0.9],
            [0.7, 0.8, 0.3],
            [0.4, 0.6, 0.2],
            [0.3, 0.2, 0.8],
        ]
        _OPTIMAL_PARAMS = np.asarray(_OPTIMAL_PARAMS)

        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=0.5,
                lb=0.0,
                ub=1.0,
            ),
            ContinuousValue(
                name="p2",
                initial_value=0.3,
                lb=0.0,
                ub=1.0,
            ),
            ContinuousValue(
                name="p3",
                initial_value=0.1,
                lb=0.0,
                ub=1.0,
            ),
        )

        features = [
            {
                "name": "jobtype",
                "values": [0, 1, 2, 3, 4, 5, 6, 7],
            },
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

        for seed in range(6):  # Trying multiple seeds
            tuner_dgt = OPPerTune(
                algorithm="dgt",
                parameters=parameters,
                algorithm_args=dict(
                    features=features,
                    height=3,
                    feedback=2,
                    eta1=0.01,
                    eta2=0.01,
                    delta=0.1,
                    random_seed=seed,
                    optimizer="sgd",
                    internal_weights=internal_weights,
                    fix_internal_weights=False,
                    leaf_weights=None,
                ),
            )

            num_iterations = 1000
            jobtypes = [0, 1, 2, 3, 4, 5, 6, 7]
            for _ in range(num_iterations):
                jobtype = rng.choice(jobtypes)
                cur_features = {"jobtype": jobtype}

                pred, _metadata = tuner_dgt.predict(features=cur_features)

                pred_arr = np.asarray(list(pred.values()))
                reward = get_reward(pred_arr, jobtype)

                tuner_dgt.set_reward(reward, metadata=_metadata)

            for jobtype in jobtypes:
                cur_features = {"jobtype": jobtype}

                pred, _metadata = tuner_dgt.predict(features=cur_features)
                pred_arr = np.asarray(list(pred.values()))

                reward = get_reward(pred_arr, jobtype)

                # Assuming converged if the reward is > -0.15.
                # This need not be true always especially if a jobtype does not receive enough samples.
                self.assertGreater(reward, -0.15)

    def test_step_size(self):
        """Check if predicted values respect step_size constraint."""
        parameters = (
            ContinuousValue(
                name="p1",
                initial_value=30.5,
                lb=0.25,
                ub=30.5,
                step_size=None,
            ),
            DiscreteValue(
                name="p2",
                initial_value=30,
                lb=0,
                ub=30,
                step_size=1,
            ),
            DiscreteValue(
                name="p3",
                initial_value=128,
                lb=32,
                ub=256,
                step_size=32,
            ),
            ContinuousValue(
                name="p4",
                initial_value=32.0,
                lb=2.0,
                ub=64.0,
                step_size=2.0,
            ),
            DiscreteValue(
                name="p5",
                initial_value=65536,
                lb=536,
                ub=65536,
                step_size=1000,
            ),
            DiscreteValue(
                name="p6",
                initial_value=49600,
                lb=19600,
                ub=49600,
                step_size=10000,
            ),
            DiscreteValue(
                name="p7",
                initial_value=5,
                lb=3,
                ub=10,
                step_size=1,
            ),
            ContinuousValue(
                name="p8",
                initial_value=5.0,
                lb=3.0,
                ub=10.0,
            ),
        )

        in_features = 5
        features = [
            {
                "name": "f",
                "values": tuple(range(in_features)),
            },
        ]

        tuner_dgt = OPPerTune(
            algorithm="dgt",
            parameters=parameters,
            algorithm_args={
                "features": features,
                "height": 3,
                "feedback": 2,
                "eta1": 0.001,
                "eta2": 1,
                "delta": 1,
                "random_seed": 123,
                "optimizer": "rmsprop",
                "internal_weights": [
                    [1, 1, 1, 1, -1],
                    [1, 1, -1, -1, 0],
                    [0, 0, 0, 0, 1],
                    [1, -1, 0, 0, 0],
                    [0, 0, 1, -1, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                ],
                "fix_internal_weights": False,
                "leaf_weights": None,
            },
        )

        for _ in range(10):
            for i in range(in_features):
                cur_features = {"f": i}

                pred, _metadata = tuner_dgt.predict(features=cur_features)
                tuner_dgt.set_reward(reward=np.random.random(), metadata=_metadata)  # Dummy reward

                for p in parameters:
                    pred_value = pred[p.name]

                    if p.step_size is not None:
                        assert (
                            (p.lb - pred_value) / p.step_size
                        ).is_integer(), f"[{p.name}] Invalid predicted value, not reachable using lb={p.lb}"

                        assert (
                            (p.ub - pred_value) / p.step_size
                        ).is_integer(), f"[{p.name}] Invalid predicted value, not reachable using ub={p.ub}"
