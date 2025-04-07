"""The AutoScope algorithm."""

import abc
import warnings
from dataclasses import dataclass
from typing import Any, Generic, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from numpy import typing as npt
from torch import nn, optim
from typing_extensions import Literal, TypedDict, TypeVar, override

from oppertune.core.types import Context, PredictResponse
from oppertune.core.values import Integer, Real

from ..base import Algorithm, _PredictResponse, _TuningRequest
from ..utils.common import torch_to_numpy_dtype
from ..utils.vector import unit_normal_vector as _unit_normal_vector
from .utils import ScaleBinarizer, Sparser, XLinear

__all__ = (
    "AutoScope",
    "IAutoScopeLeaf",
    "AutoScopeFeature",
)

_ParameterValueType = TypeVar("_ParameterValueType", bound=Union[int, float], default=Union[int, float])


def unit_normal_vector_like(v: torch.Tensor, generator: np.random.Generator) -> torch.Tensor:
    dtype = torch_to_numpy_dtype(v.dtype)
    vec = _unit_normal_vector(tuple(v.shape), generator=generator, dtype=dtype)
    return torch.from_numpy(vec)


class AutoScopeFeature(TypedDict):
    name: str
    values: Union[Sequence[str], None]


@dataclass
class Metadata:
    selected_leaf: int


AutoScopeFeatureMap = Mapping[str, Union[str, int, float]]  # E.g., {"f1": "large", "f2": 1, "f3": 16.5}


class IAutoScopeLeaf(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def autoscope_gradient(self) -> Union[torch.Tensor, None]:
        """Returns the gradient of the leaf algorithm with respect to the reward."""
        grad = self._autoscope_gradient()
        if grad is None or isinstance(grad, torch.Tensor):
            return grad

        return torch.tensor(grad)

    @property
    def autoscope_center(self) -> torch.Tensor:
        """Returns the current optimal center of the leaf algorithm."""
        center = self._autoscope_center()
        if isinstance(center, torch.Tensor):
            return center

        return torch.tensor(center)

    @abc.abstractmethod
    def _autoscope_gradient(self) -> Union[torch.Tensor, npt.NDArray, Sequence, np.generic, float, None]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _autoscope_center(self) -> Union[torch.Tensor, npt.NDArray, Sequence, np.generic, float, None]:
        raise NotImplementedError()


class AutoScope(Algorithm, nn.Module, Generic[_ParameterValueType]):
    class Meta:
        supported_parameter_types = (Integer, Real)
        requires_untransformed_parameters = False
        supports_context = True
        supports_single_reward = True
        supports_sequence_of_rewards = False

    def __init__(
        self,
        parameters: Iterable[Union[Integer, Real]],
        features: Iterable[AutoScopeFeature],
        leaf_algorithm: str = "bluefin",
        leaf_algorithm_args: Optional[Mapping[str, Any]] = None,
        height: int = 3,
        and_bias_is_learnable: bool = False,
        over_param: Optional[Sequence[int]] = None,
        eta: float = 0.01,
        delta: float = 0.1,
        optimizer: Literal["rmsprop", "sgd"] = "rmsprop",
        optimizer_args: Optional[Mapping[str, Any]] = None,
        internal_weights: Optional[List] = None,
        fix_internal_weights: bool = False,
        random_seed: Optional[int] = None,
    ):
        super().__init__(parameters, random_seed=random_seed)
        self.params: Tuple[Union[Integer, Real], ...]  # For type hints

        self.features = tuple(features)
        self.height = height

        # Algorithm hyperparameters
        self.eta = eta
        self.delta = delta

        # Setting up random number generator
        # We are not using torch.Generator() because it is not pickle-able
        self.rng = np.random.default_rng(random_seed)

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
        for feature in self.features:
            name = feature["name"]
            values = feature["values"]
            if values is None:  # Numerical feature
                counter += 1
            else:
                if len(values) <= 1:
                    raise ValueError(f"Categorical feature {name} has less than 2 values")

                counter += len(values)

        self._in_features = counter
        self._out_features = len(self.params)

        self._and_act_fn = nn.Softmax(dim=-1)

        n_leaf_nodes = 2**self.height
        n_internal_nodes = n_leaf_nodes - 1

        # L1 layer / Predicate layer
        self._over_param = tuple(over_param) if over_param else None  # TODO Remove
        _predicate_layers: List[nn.Linear] = []
        dtype = torch.float64
        if self._over_param is None:  # No over-parameterization
            _predicate_layers.append(
                nn.Linear(in_features=self._in_features, out_features=n_internal_nodes, dtype=dtype),
            )
        else:
            # With over-parameterization, we learn a deep representation of the features, before it is fed to the
            # decision tree. We do this by introducing fully-connected hidden layers between the input and the
            # decision tree.
            warnings.warn("over_param is deprecated and will be removed in subsequent versions", stacklevel=1)
            n_over_param_nodes_in_layer = (
                self._in_features,
                *(n_op * n_internal_nodes for n_op in self._over_param),
                n_internal_nodes,  # The final layer will connect to the internal nodes
            )

            for i in range(len(n_over_param_nodes_in_layer) - 1):
                _predicate_layers.append(
                    nn.Linear(
                        in_features=n_over_param_nodes_in_layer[i],
                        out_features=n_over_param_nodes_in_layer[i + 1],
                        dtype=dtype,
                    ),
                )

        # Initializing with zero bias
        for layer in _predicate_layers:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.bias)

        self._predicate_layers = nn.Sequential(*_predicate_layers)

        if internal_weights is not None:
            assert self._over_param is None, "Cannot provide internal weights when over-parameterization is used"
            self._predicate_layers[0].weight = nn.Parameter(torch.tensor(internal_weights, dtype=dtype))

        if fix_internal_weights:
            self._predicate_layers[0].requires_grad_(False)

        # L2 layer / And layer
        weight, fixed_bias = AutoScope._get_and_layer_params(height, dtype=dtype)
        self._and_layer = XLinear(
            n_internal_nodes,
            n_leaf_nodes,
            weight=weight,
            bias=None if and_bias_is_learnable else fixed_bias,
            same=False,
            dtype=dtype,
        )

        # Optimizer
        if optimizer_args is None:
            optimizer_args = {"lr": eta}

        if optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self._predicate_layers.parameters(), **optimizer_args)
        else:
            self.optimizer = optim.SGD(self._predicate_layers.parameters(), **optimizer_args)

        from ..all import create_tuning_instance, get_algorithm_class  # To avoid circular import

        _algorithm_class = get_algorithm_class(leaf_algorithm)
        if not issubclass(_algorithm_class, IAutoScopeLeaf):
            raise TypeError(f"{leaf_algorithm} is not a valid leaf algorithm for {self.__class__.__name__}")

        if _algorithm_class.Meta.supports_context:
            raise TypeError("Leaf algorithm must not be contextful")

        self.leaf_tuning_instances = tuple(
            create_tuning_instance(self.params, leaf_algorithm, leaf_algorithm_args) for _ in range(n_leaf_nodes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO More descriptive names
        # L1 layer / Predicate layer
        pred_z: torch.Tensor = self._predicate_layers(x)

        pred_a, fac = ScaleBinarizer.apply(pred_z)
        fac = fac.detach()

        pred_a = 2 * pred_a - fac

        # L2 layer / And layer
        and_z_a = self._and_layer(pred_a)

        and_a = self._and_act_fn(and_z_a)
        selected_leaf_vector: torch.Tensor = Sparser.apply(and_a)
        return selected_leaf_vector

    @override
    def predict(self, context: Context, predict_data: None = None) -> PredictResponse[_ParameterValueType]:
        return super().predict(context, predict_data)

    @override
    def _predict(self, context: Context, predict_data: None = None) -> _PredictResponse[_ParameterValueType]:
        feature_map = context.data
        if feature_map is None:
            raise ValueError("A valid context is required")

        features: torch.Tensor = self._get_feature_vector(feature_map)
        selected_leaf_vector: torch.Tensor = self(features)
        selected_leaf = int(selected_leaf_vector.argmax())
        parameters, _ = self.leaf_tuning_instances[selected_leaf]._predict(context=None)
        return _PredictResponse(parameters, metadata=Metadata(selected_leaf))

    @override
    def _set_reward(self, tuning_request: _TuningRequest[_ParameterValueType]) -> None:
        feature_map = tuning_request.context.data
        if feature_map is None:
            raise ValueError("A valid context is required")

        features: torch.Tensor = self._get_feature_vector(feature_map)

        # Check if selected_leaf has changed. In case the selected leaf has changed,
        # we cannot update the weights since the reward corresponds to a different leaf.
        selected_leaf_vector: torch.Tensor = self(features)
        current_selected_leaf = int(selected_leaf_vector.argmax())
        metadata: Metadata = tuning_request.metadata
        if current_selected_leaf != metadata.selected_leaf:
            return

        selected_leaf_tuning_instance = self.leaf_tuning_instances[current_selected_leaf]
        assert isinstance(selected_leaf_tuning_instance, IAutoScopeLeaf)

        # Get all predictions from leaves. This is equivalent to constructing the OR layer weight.
        all_predictions = torch.stack([leaf.autoscope_center for leaf in self.leaf_tuning_instances])  # type: ignore

        selected_leaf_tuning_instance._set_reward(tuning_request)
        leaf_grad = selected_leaf_tuning_instance.autoscope_gradient
        if leaf_grad is None:
            return

        leaf_grad *= -1.0  # Multiplying by -1 to get the loss gradient
        leaf_grad = leaf_grad.reshape((1, -1))

        # Get the gradient for the and_layer
        grad_vector = leaf_grad @ all_predictions.T

        self.optimizer.zero_grad()
        selected_leaf_vector.requires_grad_(True)  # Output of Sparser has grad set to False
        selected_leaf_vector.backward(grad_vector)
        self.optimizer.step()

    def _get_feature_vector(self, features: AutoScopeFeatureMap) -> torch.Tensor:
        """Utility function to convert dictionary features to a feature tensor.

        Args:
            features: A dictionary indicating the features to activate in the feature tensor.

        Returns:
            The feature tensor.
        """
        feature_tensor = torch.zeros(size=(self._in_features,), dtype=self._and_layer.linear.weight.dtype)

        feature_idx = 0
        for feature in self.features:
            name = feature["name"]
            values = feature["values"]
            val = features[name]

            if values is None:  # Numerical feature, expecting floating type
                if not isinstance(val, (int, float)):
                    raise TypeError(f"Expected numeric value for feature {name}")

                feature_tensor[feature_idx] = val
                feature_idx += 1
            else:  # Categorical feature
                if not isinstance(val, str):
                    raise TypeError(f"Expected categorical value for feature {name}")

                feature_tensor[feature_idx + values.index(val)] = 1.0
                feature_idx += len(values)

        return feature_tensor.reshape((1, -1))

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
        n_nodes_in_cur_level = 1
        n_leaf_nodes = 2**height
        n_internal_nodes = n_leaf_nodes - 1

        weight = torch.zeros((n_leaf_nodes, n_internal_nodes), dtype=dtype)

        # Fill in the weight matrix level by level
        for _ in range(height):
            start_idx = n_nodes_in_cur_level - 1  # Index of the first node in this level

            row_len = n_leaf_nodes // n_nodes_in_cur_level
            row_mid_len = row_len // 2
            for idx in range(start_idx, start_idx + n_nodes_in_cur_level):  # Iterate through nodes at this level
                row_begin = row_len * (idx - start_idx)
                row_mid = row_begin + row_mid_len
                row_end = row_begin + row_len

                weight[row_begin:row_mid, idx] = 1
                weight[row_mid:row_end, idx] = -1

            n_nodes_in_cur_level *= 2  # The next level will have 2 times the number of nodes

        fixed_bias = torch.zeros(size=(n_leaf_nodes,), dtype=dtype)
        return weight, fixed_bias

    @property
    @override
    def iteration(self) -> int:
        return self._iteration
