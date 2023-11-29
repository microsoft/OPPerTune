from hashlib import sha256
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing_extensions import Literal, TypedDict

from ...normalizers import min_max
from ...utils.common import torch_to_numpy_dtype
from ...values import ContinuousValue, DiscreteValue
from ...vectors import unit_normal_vector as _unit_normal_vector
from ..base import AlgorithmBackend, PredictResponse
from .dgt_utils import ScaleBinarizer, Sparser, XLinear

__all__ = ("DGT",)


def unit_normal_vector_like(v: torch.Tensor, generator: np.random.Generator) -> torch.Tensor:
    dtype = torch_to_numpy_dtype(v.dtype)
    vec = _unit_normal_vector(tuple(v.shape), generator=generator, dtype=dtype)
    return torch.from_numpy(vec)


def hash_tensor(tensor: torch.Tensor) -> str:
    tensor_string = ",".join(map(str, tensor.tolist()))
    return sha256(tensor_string.encode("utf-8")).hexdigest()


class FIDMap(TypedDict):
    w_plus: torch.Tensor
    w_minus: torch.Tensor
    u: torch.Tensor
    first_reward: Union[float, None]
    second_reward: Union[float, None]
    features: torch.Tensor
    selected_leaf: int


class Metadata(TypedDict):
    selected_leaf: int
    features: List[float]
    f_id: str


class Feature(TypedDict):
    name: str
    values: Union[Sequence[Union[bool, float, int, str]], None]


FeaturesDict = Dict[str, Union[bool, float, int, str]]  # E.g., {"f1": True, "f2": "large", "f3": 16.5, "f4": 1}


class DGT(nn.Module, AlgorithmBackend):
    def __init__(
        self,
        parameters: Iterable[Union[ContinuousValue, DiscreteValue]],
        features: Iterable[Feature],
        height: int = 3,
        and_bias_is_learnable: bool = False,
        or_bias_is_learnable: bool = False,
        over_param: Optional[Tuple[int, ...]] = None,
        feedback: Literal[1, 2] = 1,
        eta1: float = 0.01,
        eta2: float = 0.01,
        delta: float = 0.1,
        optimizer: Optional[Literal["rmsprop", "sgd"]] = None,
        optimizer_kwargs: Optional[dict] = None,
        random_seed=None,
        internal_weights: Optional[List] = None,
        fix_internal_weights: bool = False,
        leaf_weights: Optional[Sequence[Sequence[float]]] = None,
    ):
        super().__init__()

        self.float_dtype = torch.float64

        # Algorithm hyperparameters
        self.eta1 = eta1
        self.eta2 = eta2
        self.delta = delta
        assert feedback in (1, 2)
        self.feedback = feedback
        self.prev_reward: Union[float, int] = 0
        self._round: int = 0

        # Setting up parameters to tune
        self.params = tuple(parameters)
        assert len(self.params) >= 1
        w_center = torch.tensor(data=[p.initial_value for p in parameters], dtype=self.float_dtype)
        lb = torch.tensor(data=[p.lb for p in parameters], dtype=self.float_dtype)
        ub = torch.tensor(data=[p.ub for p in parameters], dtype=self.float_dtype)
        step_size = torch.tensor(
            [torch.nan if p.step_size is None else p.step_size for p in parameters], dtype=self.float_dtype
        )
        is_discrete = torch.tensor([isinstance(p, DiscreteValue) for p in parameters], dtype=self.float_dtype)

        # Divide by step_size to make step size as 1 to make calculations simpler
        mask = ~torch.isnan(step_size)
        step_size_mask = step_size[mask]
        lb[mask] /= step_size_mask
        ub[mask] /= step_size_mask
        w_center[mask] /= step_size_mask

        w_center = min_max.normalize_(w_center, lb, ub)

        self.lb_normalized, self.ub_normalized = 0.0, 1.0

        self.lb = lb
        self.ub = ub
        self.step_size = step_size
        self.is_discrete = is_discrete

        # Setting up random number generator
        # We are not using torch.Generator() because it is not pickle-able
        self.rng = np.random.default_rng(random_seed)

        self.feature_list = tuple(features)
        self.feature_map = {}
        self.fid_map = {}

        # Assign indices to each feature. If the feature is numerical, it gets a single index. If the feature is
        # categorical, it gets one-hot encoded and the assigned index in feature_map is the starting index of the
        # one-hot encoding. For example, if
        # features = [
        #     {
        #         "name": "f1", # Numerical feature
        #         "values": None
        #     },
        #     {
        #         "name": "f2", # Categorical feature
        #         "values": ("a", "b", "c", "d", "e")
        #     },
        #     {
        #         "name": "f3", # Categorical feature
        #         "values": (True, False)
        #     },
        #     {
        #        "name": "f4", # Numerical feature
        #        "values": None
        #     }
        # ]
        # then feature_map = {"f1": 0, "f2": 1, "f3": 6, "f4": 8}
        counter = 0
        for f in self.feature_list:
            name = f["name"]
            values = f["values"]
            self.feature_map[name] = counter
            if values is None:  # Numerical feature
                counter += 1
            else:
                assert len(values) > 1, f"Categorical feature {name} has less than 2 values"
                counter += len(values)

        self._in_features = counter
        self._out_features = len(self.params)

        self._and_act_fn = nn.Softmax(dim=-1)
        self._over_param = over_param

        num_internal_nodes = 2**height - 1
        num_leaf_nodes = 2**height

        # L1 layer / Predicate layer
        _predicate_layers: List[nn.Linear] = []

        if self._over_param is None:  # No overparameterization
            _predicate_layers.append(
                nn.Linear(in_features=self._in_features, out_features=num_internal_nodes, dtype=self.float_dtype),
            )
        else:
            # With overparameterization, we learn a deep representaion of the features, before it is fed to the
            # decision tree. We do this by introducing fully-connected hidden layers between the input and the
            # decision tree.
            n_over_param_nodes_in_layer = (
                self._in_features,
                *(n_op * num_internal_nodes for n_op in self._over_param),
                num_internal_nodes,  # The final layer will connect to the internal nodes
            )

            for i in range(len(n_over_param_nodes_in_layer) - 1):
                _predicate_layers.append(
                    nn.Linear(
                        in_features=n_over_param_nodes_in_layer[i],
                        out_features=n_over_param_nodes_in_layer[i + 1],
                        dtype=self.float_dtype,
                    ),
                )

        # Initializing with zero bias
        with torch.no_grad():
            for layer in _predicate_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.bias)

        self._predicate_layers = nn.Sequential(*_predicate_layers)

        if internal_weights:
            assert self._over_param is None, "Cannot provide internal weights when overparameterization is used"
            self._predicate_layers[0].weight = nn.Parameter(torch.tensor(internal_weights, dtype=self.float_dtype))

        if fix_internal_weights:
            for param in self._predicate_layers[0].parameters():
                param.requires_grad = False

        # L2 layer / And layer
        weight, fixed_bias = DGT._get_and_layer_params(height, dtype=self.float_dtype)
        self._and_layer = XLinear(
            num_internal_nodes,
            num_leaf_nodes,
            weight=weight,
            bias=None if and_bias_is_learnable else fixed_bias,
            same=False,
            dtype=self.float_dtype,
        )

        # L3 layer / Or layer
        self._or_layer = XLinear(
            num_leaf_nodes,
            self._out_features,
            bias=None if or_bias_is_learnable else torch.zeros(size=(self._out_features,), dtype=self.float_dtype),
            same=True,
            dtype=self.float_dtype,
        )

        # Optimizer
        if optimizer is None:
            assert optimizer_kwargs is None
            optimizer = "sgd" if feedback == 1 else "rmsprop"

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(
                [
                    {
                        "params": self.get_parameters_set(set_idx=1),
                        "lr": eta1,
                    },  # Inner nodes setting
                    {
                        "params": self.get_parameters_set(set_idx=2),
                        "lr": eta2,
                    },  # Leaf nodes setting
                ],
                **optimizer_kwargs,
            )
        else:
            self.optimizer = optim.SGD(
                [
                    {
                        "params": self.get_parameters_set(set_idx=1),
                        "lr": eta1,
                    },  # Inner nodes setting
                    {
                        "params": self.get_parameters_set(set_idx=2),
                        "lr": eta2,
                    },  # Leaf nodes setting
                ],
                **optimizer_kwargs,
            )

        if not leaf_weights:  # If leaf_weights are not provided
            # Use the initial values of the parameters as the leaf weights.
            # Each leaf node gets the same weight i.e initial value of parameters
            leaf_weights_tensor = w_center.repeat(num_leaf_nodes, 1).T
        else:
            # Use provided leaf weights
            leaf_weights_tensor = torch.tensor(leaf_weights, dtype=self.float_dtype)

        with torch.no_grad():
            self._or_layer.weight.copy_(leaf_weights_tensor)

        self.clip_center()

    @property
    def round(self):
        return self._round // self.feedback

    def get_parameters_set(self, set_idx: int) -> Iterator[nn.Parameter]:
        """Get the set of parameters for the optimizer."""
        if set_idx == 1:  # Parameters for internal nodes
            return self._predicate_layers.parameters()

        elif set_idx == 2:
            return self._or_layer.parameters()

        else:
            raise ValueError(f"{set_idx} must be in [1, 2]")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L1 layer / Predicate layer
        pred_z: torch.Tensor = self._predicate_layers(x)

        pred_a, fac = ScaleBinarizer.apply(pred_z)
        fac = fac.detach()

        pred_a = 2 * pred_a - fac

        # L2 layer / And layer
        and_z_a = self._and_layer(pred_a)

        and_a = self._and_act_fn(and_z_a)
        and_a: torch.Tensor = Sparser.apply(and_a)

        # L3 layer / Or layer
        or_z = self._or_layer(and_a)

        with torch.no_grad():
            self.selected_leaf = int(torch.argmax(and_a))

        return or_z

    def format(self, values: torch.Tensor) -> Dict[str, Union[float, int]]:
        return {param.name: param.cast(value) for param, value in zip(self.params, values.tolist())}

    def predict(self, features: FeaturesDict):
        """
        Args:
            features: A dictionary indicating the features to activate in the feature tensor.

        Returns:
            Same as oppertune.predict. metadata contains the selected leaf index, the feature tensor (as a list)
            and the feature id (f_id) which is a hash of the tensor.
        """
        _features: torch.Tensor = self.process_feature_dict(features)
        f_id = hash_tensor(_features)

        if self.fid_map.get(f_id, None) is None:
            y = self.forward(x=_features)
            w = y.reshape(shape=(-1,))

            u = unit_normal_vector_like(w, generator=self.rng)

            w_plus = DGT.compute_explore_value(
                y=w,
                explore_sign=1,
                explore_direction=u,
                delta=self.delta,
                lb=self.lb,
                ub=self.ub,
                step_size=self.step_size,
            )
            w_minus = DGT.compute_explore_value(
                y=w,
                explore_sign=-1,
                explore_direction=u,
                delta=self.delta,
                lb=self.lb,
                ub=self.ub,
                step_size=self.step_size,
            )

            self.fid_map[f_id] = FIDMap(
                w_plus=w_plus,
                w_minus=w_minus,
                u=u,
                first_reward=None,
                second_reward=None,
                features=_features,
                selected_leaf=self.selected_leaf,
            )
        else:
            assert torch.equal(
                self.fid_map[f_id]["features"], _features
            ), f"Invalid feature_id={f_id}. features={self.fid_map[f_id]['features']} != {_features}"

        if self.feedback == 1:
            values = self.fid_map[f_id]["w_plus"]
        else:
            if self.fid_map[f_id]["first_reward"] is None:
                values = self.fid_map[f_id]["w_plus"]
            else:
                values = self.fid_map[f_id]["w_minus"]

        parameters = self.format(values)

        metadata = Metadata(
            selected_leaf=self.fid_map[f_id]["selected_leaf"],
            features=_features.tolist(),
            f_id=f_id,
        )

        return PredictResponse(parameters=parameters, metadata=metadata)

    def set_reward(self, reward: float, metadata: Metadata):
        """
        Same as oppertune.set_reward.
        """
        f_id = metadata["f_id"]
        features = torch.tensor(metadata["features"], dtype=self.float_dtype)
        selected_leaf = metadata["selected_leaf"]

        # Check if selected_leaf has changed. In case the selected leaf has changed,
        # we cannot update the weights since the reward corresponds to a different leaf.
        with torch.no_grad():
            self.forward(x=features)  # This updates self.selected leaf

        if self.selected_leaf != selected_leaf:
            return None

        assert torch.equal(
            self.fid_map[f_id]["features"], features
        ), f"Invalid feature_id={f_id}. features={self.fid_map[f_id]['features']} != {features}"

        if self.feedback == 1:  # onepoint
            self.update_weights(reward, f_id, features)
            self._round += 1
            self.fid_map[f_id] = None
        else:  # twopoint
            if self.fid_map[f_id]["first_reward"] is None:
                self.fid_map[f_id]["first_reward"] = reward
            else:
                reward_diff = self.fid_map[f_id]["first_reward"] - reward

                # Normalizing reward diff
                reward_denominator = max(abs(self.fid_map[f_id]["first_reward"]), abs(reward))
                if reward_denominator != 0:
                    reward_diff /= reward_denominator

                self.update_weights(reward_diff, f_id, features)
                self._round += 1
                self.fid_map[f_id] = None

    def update_weights(self, reward_grad, f_id, features):
        self.optimizer.zero_grad()
        y = self.forward(x=features)
        with torch.no_grad():
            grad = self._out_features * reward_grad * -1 * self.fid_map[f_id]["u"] / (self.feedback * self.delta)
            grad = torch.Tensor(grad.tolist()).reshape(shape=y.shape)
        y.backward(gradient=grad)

        self.optimizer.step()
        self.clip_center()

    @torch.no_grad()
    def clip_center(self):
        self._or_layer.weight.clamp_(self.lb_normalized + self.delta, self.ub_normalized - self.delta)

    def process_feature_dict(self, features: FeaturesDict) -> torch.Tensor:
        """
        Utility function to convert dictionary features to a feature tensor.

        Args:
            features: A dictionary indicating the features to activate in the feature tensor.

        Returns:
            The feature tensor.
        """
        feature_tensor = torch.zeros(size=(self._in_features,), dtype=self.float_dtype)

        f_idx = 0
        for f in self.feature_list:
            name = f["name"]
            values = f["values"]
            if values is None:  # Numerical feature, expecting floating type
                feature_tensor[f_idx] = features[name]
                f_idx += 1
            else:  # Categorical feature
                feature_tensor[f_idx + values.index(features[name])] = 1.0
                f_idx += len(values)

        return feature_tensor.reshape(shape=(1, -1))

    @staticmethod
    def fit_to_constraints(v: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor, step_size: torch.Tensor):
        v = v.clip(lb, ub)

        mask = ~torch.isnan(step_size)
        lb_masked = lb[mask]
        step_size_masked = step_size[mask]

        n_minus_1_masked = (v[mask] - lb_masked).round()
        v[mask] = (lb_masked + n_minus_1_masked) * step_size_masked
        return v

    @staticmethod
    def compute_explore_value(
        y: torch.Tensor,
        explore_sign: int,
        explore_direction: torch.Tensor,
        delta: float,
        lb: torch.Tensor,
        ub: torch.Tensor,
        step_size: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the point to explore.

        Computes the point to explore by randomly perturbing the center vector. This explore vector is sampled from a
        hypersphere centered at the center vector with a radius of delta.

        Args:
            y: The output of the DGT model.
            explore_sign: Specifies whether exploration is in the direction of explore_dir or opposite to it.
            explore_direction: Direction of exploration.
            delta: The exploration radius.
            lb: The lower bound of all parameters in y.
            ub: The upper bound of all parameters in y.

        Returns:
            An array that represents the point to explore.
        """
        with torch.no_grad():
            change = explore_sign * delta * explore_direction
            explore_value = y + change
            explore_value = min_max.denormalize_(explore_value, lb, ub)
            explore_value = DGT.fit_to_constraints(explore_value, lb, ub, step_size)

        return explore_value

    @staticmethod
    def _get_and_layer_params(height: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the weight and bias matrix for a given height of the tree.

        This function computes the weights connecting the first layer of the network (whose nodes represent the
        internal nodes of the tree) and the second layer of the network (whose nodes represent the leaf nodes in the
        tree).
        Let's assume we are currently computing the weights for some internal node Z. Weight values for connections
        from Z to those leaves that are not part of the subtree rooted at Z should be 0. Weight values for nodes that
        are part of the left subtree rooted at Z should be 1 and -1 for the nodes that are part of the right subtree.

        Args:
            height: The height of the tree.
            dtype: The floating point type of the weights and biases.

        Returns:
            A tuple of that represents the weights and biases.
        """
        num_internal_nodes = 2**height - 1
        num_leaf_nodes = 2**height

        weight = torch.zeros((num_leaf_nodes, num_internal_nodes), dtype=dtype)

        # Fill in the weight matrix level by level
        # h represents the level of nodes which we are handling at a given iteration
        for h in range(height):
            num_nodes = 2**h  # Number of nodes in this level
            start_idx = num_nodes - 1  # Index of the first node in this level

            for idx in range(start_idx, start_idx + num_nodes):  # Iterate through all nodes at this level
                row_begin = (num_leaf_nodes // num_nodes) * (idx - start_idx)
                row_mid = row_begin + (num_leaf_nodes // (2 * num_nodes))
                row_end = row_begin + (num_leaf_nodes // num_nodes)

                weight[row_begin:row_mid, idx] = 1
                weight[row_mid:row_end, idx] = -1

        fixed_bias = torch.zeros(size=(2**height,), dtype=dtype)

        return weight, fixed_bias
