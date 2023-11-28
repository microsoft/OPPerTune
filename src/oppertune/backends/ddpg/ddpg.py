from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions.bernoulli as torchdist

from ...values import ContinuousValue, ContinuousValueDict, DiscreteValue, DiscreteValueDict
from ..base import AlgorithmBackend, PredictResponse
from .environment import ContainerEnvironment

__all__ = ("DDPG",)

SupportedValueDictTypes = Union[ContinuousValueDict, DiscreteValueDict]
SupportedValueTypes = Union[ContinuousValue, DiscreteValue]


class Actor(torch.nn.Module):
    """
    Actor network class
    Directly predicts the policy State --> Action
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_width: int, max_action: torch.Tensor):
        super().__init__()
        self.max_action = max_action
        self.l1 = torch.nn.Linear(state_dim, hidden_width, dtype=torch.float64)
        self.l2 = torch.nn.Linear(hidden_width, action_dim, dtype=torch.float64)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s = self.relu(self.l1(s))
        a = self.max_action * self.sigmoid(self.l2(s))  # [-max, max]
        return a


class Critic(torch.nn.Module):
    """
    The Critic Network class
    According to (s,a), directly calculate Q(s,a)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_width: int):
        super().__init__()
        self.l1 = torch.nn.Linear(state_dim + action_dim, hidden_width, dtype=torch.float64)
        self.l2 = torch.nn.Linear(hidden_width, 1, dtype=torch.float64)
        self.relu = torch.nn.ReLU()

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([s, a], dim=1)
        q = self.relu(self.l1(sa))
        q = self.l2(q)
        return q


class ReplayBuffer(object):
    def __init__(self, state_dim: int, action_dim: int):
        self.max_size = 1_000_000
        self.count = 0
        self.size = 0
        self.s = torch.zeros((self.max_size, state_dim), dtype=torch.float64)
        self.a = torch.zeros((self.max_size, action_dim), dtype=torch.float64)
        self.r = torch.zeros((self.max_size, 1), dtype=torch.float64)
        self.s_ = torch.zeros((self.max_size, state_dim), dtype=torch.float64)
        self.dw = torch.zeros((self.max_size, 1), dtype=torch.float64)

    def store(self, s: torch.Tensor, a: torch.Tensor, r: float, s_: torch.Tensor, dw: Any):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of transitions

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        i = torch.randperm(self.size)[:batch_size]  # Randomly sample an index
        return self.s[i], self.a[i], self.r[i], self.s_[i], self.dw[i]


class Agent(object):
    """
    The DDPG Agent
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: torch.Tensor,
        hidden_width: int = 64,
        batch_size: int = 5,
        gamma: float = 0.1,
        tau: float = 0.005,
        lr: float = 3e-4,
    ):
        self.hidden_width = hidden_width
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action)
        self.actor_target = deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, self.hidden_width)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.lr)

        self.mse_loss = torch.nn.MSELoss()

    def choose_action(self, s: torch.Tensor) -> torch.Tensor:
        s = torch.unsqueeze(s, 0)
        return self.actor(s).detach().flatten()

    def learn(self, relay_buffer: ReplayBuffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_Q = batch_r + self.gamma * (1 - batch_dw) * Q_

        # Compute the current Q and the critic loss
        current_Q = self.critic(batch_s, batch_a)
        critic_loss = self.mse_loss(target_Q, current_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()

        # Optimize the actorstate
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learn_wrapper(self, relay_buffer: ReplayBuffer, learn_loop_count: Union[int, None]):
        if learn_loop_count is None:
            learn_loop_count = 100

        for _ in range(learn_loop_count):
            self.learn(relay_buffer)


class DDPG(AlgorithmBackend):
    """
    The DDPG wrapper / helper class
    designed as an API for the DDPG agent
    """

    def __init__(
        self,
        parameters: List[SupportedValueTypes],
        state_config: List[Union[SupportedValueDictTypes, SupportedValueTypes]],
        epsilon: float = 0.1,
        hidden_width: int = 32,
        batch_size: int = 10,
        initial_random_steps: int = 10,
        gamma: float = 0.5,
        tau: float = 0.005,
        lr: float = 3e-4,
        decay: float = 0.8,
        learn_loop_count: Union[int, None] = 400,
        random_seed: Optional[int] = None,
        is_learning: bool = True,
    ):
        self.state_config = state_config
        self.action_config = parameters
        self.env = ContainerEnvironment(
            state_config=self.state_config,
            parameters=self.action_config,
            random_seed=random_seed,
        )
        self.agent = Agent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            max_action=self.env.max_action,
            hidden_width=hidden_width,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            lr=lr,
        )
        self.initial_random_steps = initial_random_steps
        self.learning_steps = 0
        self.epsilon = epsilon
        self.decay = decay
        self.replay_buffer = ReplayBuffer(self.env.state_dim, self.env.action_dim)
        self.exploration_sampler = torchdist.Bernoulli(torch.tensor([1 - self.epsilon]))
        self.learn_loop_count = learn_loop_count
        self.previous_reward = 0.0
        self._rounds = 0
        self.is_learning = is_learning

    def predict(self, state: Optional[Union[np.ndarray, list, dict]] = None):
        if self._rounds < self.initial_random_steps:
            self.env.current_action = self.env.get_action_sample()  # a
            self._rounds += 1
            parameters = self.env.pack_actions(self.env.current_action)
            return PredictResponse(parameters=parameters)

        current_state = self.env.current_state
        if state is not None:
            current_state = self.env.unpack_state(state)
            self.env.current_state = current_state

        a = self.agent.choose_action(current_state)
        self.epsilon = self.epsilon * (self.decay ** (self._rounds / 10))  # decay the epsilon
        if not self.is_learning:  # if the client decides to stop learning (set reward)
            self.env.current_action = a
        else:
            exploration_flag = self.exploration_sampler.sample()
            self.env.current_action = a if exploration_flag >= 1.0 else self.env.get_action_sample()

        self._rounds += 1
        parameters = self.env.pack_actions(self.env.current_action)
        return PredictResponse(parameters=parameters)

    def set_reward(
        self,
        reward: Union[float, int],
        next_state: Optional[Union[np.ndarray, list, dict]] = None,
        done: bool = True,
        metadata=None,
    ):
        current_reward = float(reward)
        current_state = self.env.current_state
        current_action = self.env.current_action
        _state = current_state if next_state is None else self.env.unpack_state(next_state)

        self.replay_buffer.store(current_state, current_action, current_reward, _state, done)
        self.previous_reward = current_reward
        if self.is_learning:
            self.agent.learn_wrapper(self.replay_buffer, self.learn_loop_count)
        self.env.current_state = _state
        self.learning_steps += 1
