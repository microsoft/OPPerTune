# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 12:59:39 2022

@author: mayukhdas
"""
from typing import Dict, List, Union

import numpy as np
import torch
from gymnasium import Env, spaces

from ....values import ContinuousValue, ContinuousValueDict, DiscreteValue, DiscreteValueDict, to_value

__all__ = ("ContainerEnvironment",)

SupportedValueDictTypes = Union[ContinuousValueDict, DiscreteValueDict]
SupportedValueTypes = Union[ContinuousValue, DiscreteValue]


class ContainerEnvironment(Env):
    def __init__(
        self,
        state_config: List[Union[SupportedValueDictTypes, SupportedValueTypes]],
        parameters: List[SupportedValueTypes],
        random_seed: Union[int, None],
    ):
        super().__init__()
        self.observation_space: Union[spaces.Box, None] = None
        self.action_space: Union[spaces.Box, None] = None
        self.current_state: torch.Tensor = torch.zeros(len(state_config))
        self.current_action: torch.Tensor = torch.zeros(len(parameters))
        self.state_config: List[SupportedValueTypes] = []
        self.state_dim: int = 0
        self.action_dim: int = 0
        self.state_lb: List[float] = []
        self.state_ub: List[float] = []
        self.action_lb: List[float] = []
        self.action_ub: List[float] = []
        self.action_size: List[int] = []
        self.action_init: List[float] = []
        self.max_action: torch.Tensor = torch.zeros(len(parameters))
        self.params = parameters
        self.action_param_mapping: dict = {}
        self.space_random_seed = random_seed
        self.parse_state_action_space(state_config, parameters)

    def parse_state_action_space(
        self,
        state_config: List[Union[SupportedValueDictTypes, SupportedValueTypes]],
        parameters: List[SupportedValueTypes],
    ):
        _state_config: List[SupportedValueTypes] = [to_value(cfg) for cfg in state_config]
        self.state_config: List[SupportedValueTypes] = []
        self.state_dim = len(state_config)
        self.action_dim = len(parameters)
        self.state_lb = []
        self.state_ub = []
        self.action_lb = []
        self.action_ub = []
        self.state_init_values = []
        for v in _state_config:
            if isinstance(v, DiscreteValue):
                size = round((v.ub - v.lb) / v.step_size)
                self.state_ub.append(float(size))
            elif isinstance(v, ContinuousValue):
                self.state_ub.append(v.ub)
            else:
                raise TypeError(f"Invalid type for v: {type(v)}")

            self.state_lb.append(0.0)
            self.state_init_values.append(v.initial_value)
            self.state_config.append(v)

        self.observation_space = spaces.Box(
            low=np.array(self.state_lb),
            high=np.array(self.state_ub),
            shape=(len(self.state_lb),),
            dtype=np.float64,  # np.float64
        )

        self.action_param_mapping = {}
        action_index = 0
        for i, p in enumerate(parameters):
            if isinstance(p, ContinuousValue):
                self.action_ub.append(1.0)
                self.action_lb.append(0.0)
                self.action_size.append(1)
                self.action_init.append(self.scale(p.initial_value, p.lb, p.ub, 0.0, 1.0))
                self.action_param_mapping[i] = action_index
                action_index += 1
            elif isinstance(p, DiscreteValue):
                size = round((p.ub - p.lb) / p.step_size)
                self.action_ub.append(1.0)
                self.action_lb.append(0.0)
                self.action_size.append(size)
                self.action_init.append(self.scale(((p.initial_value - p.lb) / p.step_size), 0.0, size, 0.0, 1.0))
                self.action_param_mapping[i] = action_index
                action_index += 1
            else:
                raise TypeError(f"Invalid type for p: {type(p)}")

        self.action_space = spaces.Box(
            low=np.zeros(len(self.action_lb)),
            high=np.ones(len(self.action_ub)),
            shape=(len(self.action_lb),),
            dtype=np.float64,  # np.float64?
        )

        if self.space_random_seed is not None:  # Seeding if available
            self.action_space.seed(self.space_random_seed)
            self.observation_space.seed(self.space_random_seed)

        self.max_action = torch.tensor(self.action_ub.copy(), dtype=torch.float64)
        self.current_state = self.unpack_state(np.array([self.state_init_values]))
        self.action_dim = len(self.action_ub)

    def scale(self, value: float, old_lb: float, old_ub: float, new_lb: float, new_ub: float) -> float:
        return (((value - old_lb) * (new_ub - new_lb)) / (old_ub - old_lb)) + new_lb

    def pack_actions(self, action: Union[np.ndarray, torch.Tensor]) -> Dict[str, Union[float, int]]:
        _action: list = action.tolist()
        action_dict: Dict[str, Union[float, int]] = {}
        for i, p in enumerate(self.params):
            if isinstance(p, DiscreteValue):
                action_idx = self.action_param_mapping[i]
                action_dict[p.name] = (
                    round(self.scale(_action[action_idx], 0.0, 1.0, 0.0, self.action_size[action_idx])) * p.step_size
                ) + p.lb
            elif isinstance(p, ContinuousValue):
                action_idx = self.action_param_mapping[i]
                _num = (_action[action_idx] - self.action_lb[action_idx]) * (p.ub - p.lb)
                _den = self.action_ub[action_idx] - self.action_lb[action_idx]
                action_dict[p.name] = (_num / _den) + p.lb
            else:
                raise TypeError(f"Invalid type for p: {type(p)}")
        return action_dict

    def unpack_state(
        self, state: Union[List[Union[float, int]], Dict[str, Union[float, int]], np.ndarray]
    ) -> torch.Tensor:
        if isinstance(state, dict):
            state = [state[f.name] for f in self.state_config]

        for i, f in enumerate(state):
            v = self.state_config[i]
            if isinstance(v, DiscreteValue):
                state[i] = round((f - v.lb) / v.step_size) / self.state_ub[i]
            elif isinstance(v, ContinuousValue):
                state[i] = (f - v.lb) / (v.ub - v.lb)
            else:
                raise TypeError(f"Invalid type for v: {type(v)}")

        return torch.tensor(state, dtype=torch.float64)

    def get_action_sample(self) -> torch.Tensor:
        return torch.tensor(self.action_space.sample())

    def get_action_init(self) -> torch.Tensor:
        return torch.tensor(self.action_init)
