from copy import deepcopy
from math import sqrt
from typing import Iterable, Optional, Union

import numpy as np
from typing_extensions import Literal

from ...normalizers import min_max
from ...optimizers import SGD, RMSprop
from ...values import ContinuousValue, DiscreteValue
from ...vectors import unit_normal_vector_like
from ..base import AlgorithmBackend, PredictResponse

__all__ = ("BlueFin",)


class BlueFin(AlgorithmBackend):
    def __init__(
        self,
        parameters: Iterable[Union[ContinuousValue, DiscreteValue]],
        feedback: Literal[1, 2] = 1,
        eta: float = 0.01,
        delta: float = 0.1,
        eta_decay_rate: float = 0,
        normalize: bool = True,
        optimizer: Optional[Literal["rmsprop", "sgd"]] = None,
        optimizer_kwargs: Optional[dict] = None,
        random_seed=None,
    ):
        super().__init__()

        assert eta >= 0.0, "eta must be non-negative"
        assert delta > 0.0, "delta must be positive"

        self.params = tuple(parameters)
        assert len(self.params) >= 1
        self.eta = eta
        self.eta_initial = eta
        self.eta_decay_rate = eta_decay_rate
        self.delta = delta
        assert feedback in (1, 2)
        self.feedback = feedback
        self.normalize = normalize

        if optimizer is None:
            assert optimizer_kwargs is None
            optimizer = "sgd" if feedback == 1 else "rmsprop"

        if optimizer_kwargs is None:
            optimizer_kwargs = dict(eta=self.eta)

        self.optimizer = RMSprop(**optimizer_kwargs) if optimizer == "rmsprop" else SGD(**optimizer_kwargs)
        self.rng = np.random.default_rng(random_seed)
        self.prev_reward: Union[float, int] = 0
        self._round: int = 0

        float_dtype = np.float64
        w_center = np.array([p.initial_value for p in parameters], dtype=float_dtype)
        lb = np.array([p.lb for p in parameters], dtype=float_dtype)
        ub = np.array([p.ub for p in parameters], dtype=float_dtype)
        step_size = np.array([np.nan if p.step_size is None else p.step_size for p in parameters], dtype=float_dtype)
        is_discrete = np.array([isinstance(p, DiscreteValue) for p in parameters])

        # Divide by step_size to make step size as 1 to make calculations simpler
        mask = ~np.isnan(step_size)
        step_size_mask = step_size[mask]
        lb[mask] /= step_size_mask
        ub[mask] /= step_size_mask
        w_center[mask] /= step_size_mask

        if self.normalize:
            w_center = min_max.normalize_(w_center, lb, ub)
            self.lb_normalized, self.ub_normalized = np.zeros_like(lb), np.ones_like(ub)
        else:
            self.lb_normalized, self.ub_normalized = lb, ub

        lb_normalized_clipped, ub_normalized_clipped = self.lb_normalized + self.delta, self.ub_normalized - self.delta
        assert (lb_normalized_clipped <= ub_normalized_clipped).all(), "delta is too large. Select a smaller delta."

        if self.eta != 0:  # If eta is 0 (no learning), then we stick to the initial values of the parameters
            w_center = w_center.clip(lb_normalized_clipped, ub_normalized_clipped)

        u: np.ndarray[float_dtype] = unit_normal_vector_like(w_center, generator=self.rng)  # Perturbation direction
        delta_u = self.delta * u  # Perturbation
        w_plus = w_center + delta_u
        w_minus = w_center - delta_u

        self.w_center = w_center
        self.w_plus = w_plus
        self.w_minus = w_minus
        self.u = u
        self.lb = lb
        self.ub = ub
        self.step_size = step_size
        self.is_discrete = is_discrete

    def forward(self) -> np.ndarray:
        w = deepcopy(self.w)
        if self.normalize:
            w = min_max.denormalize_(w, self.lb, self.ub)

        w = self.fit_to_constraints(w, self.lb, self.ub, self.step_size)
        w[self.is_discrete] = w[self.is_discrete].round()
        return w

    def format(self, values: Iterable):
        return {param.name: param.cast(value) for param, value in zip(self.params, values)}

    def predict(self):
        """
        Same as oppertune.predict. metadata is empty.
        """
        values = self.forward().tolist()
        parameters = self.format(values)
        return PredictResponse(parameters=parameters)

    def set_reward(self, reward: Union[float, int], metadata=None):
        """
        Same as oppertune.set_reward.
        """
        if self.eta != 0 and self.delta != 0:
            if self.feedback == 1 or self._round % 2 == 1:
                self.__set_reward(reward)
                self._do_eta_decay()

        self.prev_reward = reward
        self._round += 1

    def __set_reward(self, reward: Union[float, int]):
        if self.feedback == 1:
            reward_diff = reward
        else:
            reward_diff = self.prev_reward - reward  # reward for w_plus - reward for w_minus
            reward_norm = max(abs(self.prev_reward), abs(reward))
            if reward_norm:
                reward_diff /= reward_norm

        grad = (self.w_center.size * reward_diff * self.u) / (self.feedback * self.delta)
        self.w_center += self.optimizer.get_step_value(grad)
        lb_normalized_clipped, ub_normalized_clipped = self.lb_normalized + self.delta, self.ub_normalized - self.delta
        assert (lb_normalized_clipped <= ub_normalized_clipped).all(), "delta is too large. Select a smaller delta."
        self.w_center = self.w_center.clip(lb_normalized_clipped, ub_normalized_clipped)

        self.u = unit_normal_vector_like(self.w_center, generator=self.rng)
        delta_u = self.delta * self.u
        self.w_plus = self.w_center + delta_u
        self.w_minus = self.w_center - delta_u

    @staticmethod
    def fit_to_constraints(v: np.ndarray, lb: np.ndarray, ub: np.ndarray, step_size: np.ndarray):
        v = v.clip(lb, ub)

        mask = ~np.isnan(step_size)
        lb_masked = lb[mask]
        step_size_masked = step_size[mask]

        n_minus_1_masked = (v[mask] - lb_masked).round()
        v[mask] = (lb_masked + n_minus_1_masked) * step_size_masked
        return v

    def _do_eta_decay(self):
        """Decay eta (and delta) based on the eta_decay_rate and current round."""
        self.eta = self.eta_initial / (1 + self.eta_decay_rate * (self.round + 1))
        self.optimizer.eta = self.eta
        self.delta = sqrt(self.eta)

    @property
    def w(self):
        if self.eta != 0:
            return self.w_plus if self.at_first_perturbation else self.w_minus
        else:  # If eta is 0 (no learning), then we stick to the initial values of the parameters
            return self.w_center

    @property
    def round(self):
        return self._round // self.feedback

    @property
    def at_first_perturbation(self):
        return self.feedback == 1 or self._round % 2 == 0

    @property
    def at_second_perturbation(self):
        return not self.at_first_perturbation
