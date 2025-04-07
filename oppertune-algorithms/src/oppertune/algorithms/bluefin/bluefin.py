"""The BlueFin algorithm."""

from math import sqrt
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
from numpy import typing as npt
from typing_extensions import Literal, TypeVar, override

from oppertune.core.types import PredictResponse
from oppertune.core.values import Integer, Real

from ..autoscope import IAutoScopeLeaf
from ..base import Algorithm, _PredictResponse, _TuningRequest
from ..utils.optimizers import SGD, RMSprop
from ..utils.vector import unit_normal_vector_like

__all__ = ("BlueFin",)

_ParameterValueType = TypeVar("_ParameterValueType", bound=Union[int, float], default=Union[int, float])


class BlueFin(Algorithm[_ParameterValueType], IAutoScopeLeaf):
    class Meta:
        supported_parameter_types = (Integer, Real)
        requires_untransformed_parameters = False
        supports_context = False
        supports_single_reward = True
        supports_sequence_of_rewards = False

    def __init__(
        self,
        parameters: Iterable[Union[Integer, Real]],
        eta: float = 0.01,
        delta: float = 0.1,
        eta_decay_rate: float = 0,
        feedback: Literal[1, 2] = 1,
        optimizer: Literal["rmsprop", "sgd"] = "rmsprop",
        optimizer_args: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
    ):
        super().__init__(parameters, random_seed=random_seed)
        self.params: Tuple[Union[Integer, Real], ...]  # For type hints

        if feedback not in (1, 2):
            raise ValueError("feedback must be either 1 or 2")

        if eta <= 0.0:
            raise ValueError("eta must be positive")

        if eta_decay_rate < 0.0:
            raise ValueError("eta_decay_rate must be non-negative")

        if delta <= 0.0:
            raise ValueError("delta must be positive")

        if 0 + delta > 1 - delta:
            raise ValueError(f"delta ({delta}) is too large. delta should be less than 0.5.")

        if optimizer not in ("rmsprop", "sgd"):
            raise ValueError(f"Invalid optimizer {optimizer!r}")

        if optimizer_args is None:
            optimizer_args = {"eta": eta}

        self.feedback = feedback
        self.eta = eta
        self._eta_initial = eta
        self.eta_decay_rate = eta_decay_rate
        self.delta = delta
        self.optimizer = RMSprop(**optimizer_args) if optimizer == "rmsprop" else SGD(**optimizer_args)
        self.reward_w_plus: float = 0
        dtype = np.float64

        _min: npt.NDArray[dtype] = np.array([p.min for p in self.params], dtype=dtype)
        assert not _min.any(), "min for all parameters must be 0"

        self._max: npt.NDArray[dtype] = np.array([p.max for p in self.params], dtype=dtype)
        is_integer: npt.NDArray[np.bool_] = np.array([isinstance(p, Integer) for p in self.params], dtype=bool)
        assert (self._max[~is_integer] == 1).all(), "max for real parameters must be 1"

        self.w_center: npt.NDArray[dtype] = np.array([p.val for p in self.params], dtype=dtype)
        self.w_center /= self._max  # Equivalent to: self.w_center[is_integer] /= self._max[is_integer]
        self.w_center = self.w_center.clip(0 + self.delta, 1 - self.delta)  # w_center +- delta * u is within [0, 1]

        self._rng = np.random.default_rng(self._random_seed)
        self.u: npt.NDArray[dtype] = unit_normal_vector_like(self.w_center, self._rng)  # Perturbation direction

        self._gradient: Union[npt.NDArray[dtype], None] = None

    @override
    def predict(self, context: None = None, predict_data: None = None) -> PredictResponse[_ParameterValueType]:
        return super().predict(context, predict_data)

    @override
    def _predict(self, context: None = None, predict_data: None = None) -> _PredictResponse[_ParameterValueType]:
        w = self.w * self._max  # Scale each parameter back to its original range [0, max]
        prediction = {param.name: param.cast(value) for param, value in zip(self.params, w.tolist())}
        return _PredictResponse(prediction)

    @override
    def _set_reward(self, tuning_request: _TuningRequest[_ParameterValueType]) -> None:
        reward = tuning_request.reward
        if reward is None:
            raise ValueError(f"reward cannot be None for {self.__class__.__name__}")

        if self.feedback == 1:
            reward_diff = reward
        else:  # if self.feedback == 2:
            if self._iteration % 2 == 0:  # First perturbation
                self.reward_w_plus = reward
                self._iteration += 1
                return

            # Second perturbation
            reward_diff = self.reward_w_plus - reward  # reward for w_plus - reward for w_minus
            reward_norm = max(abs(self.reward_w_plus), abs(reward))
            if reward_norm:
                reward_diff /= reward_norm

        self._gradient = (self.w_center.size * reward_diff * self.u) / (self.feedback * self.delta)
        self.w_center += self.optimizer.get_step_value(self._gradient)
        assert 0 + self.delta <= 1 - self.delta, f"delta ({self.delta}) is too large. delta should be less than 0.5."
        self.w_center = self.w_center.clip(0 + self.delta, 1 - self.delta)
        self.u = unit_normal_vector_like(self.w_center, self._rng)

        self._decay_eta()
        self._iteration += 1

    def _decay_eta(self) -> None:
        """Decay `eta` and `delta` based on `eta_decay_rate` and `iteration`.

        If `eta_decay_rate` is 0, then `eta` and `delta` remain unchanged.
        """
        if self.eta_decay_rate > 0:
            self.eta = self._eta_initial / (1 + self.eta_decay_rate * (self.iteration + 1))
            self.optimizer.eta = self.eta
            self.delta = sqrt(self.eta)

    @property
    def at_first_perturbation(self) -> bool:
        """Check if the algorithm is at the first perturbation.

        If `feedback` is 1, it always returns `True`.
        """
        return self.feedback == 1 or self._iteration % 2 == 0

    @property
    def w_plus(self) -> npt.NDArray[np.float64]:
        """The first perturbation vector."""
        return self.w_center + self.delta * self.u

    @property
    def w_minus(self) -> npt.NDArray[np.float64]:
        """The second perturbation vector. Only applicable when `feedback` is 2."""
        return self.w_center - self.delta * self.u

    @property
    def w(self) -> npt.NDArray[np.float64]:
        """The current perturbation vector."""
        return self.w_plus if self.at_first_perturbation else self.w_minus

    @override
    def _autoscope_gradient(self) -> Union[npt.NDArray[np.float64], None]:
        return self._gradient if self.at_first_perturbation else None

    @override
    def _autoscope_center(self) -> npt.NDArray[np.float64]:
        return self.w_center

    @property
    @override
    def iteration(self) -> int:
        return self._iteration // self.feedback
