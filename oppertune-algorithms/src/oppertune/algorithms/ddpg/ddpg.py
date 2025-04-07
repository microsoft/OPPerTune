from copy import deepcopy
from random import Random
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from gymnasium import Env, spaces
from typing_extensions import TypedDict, TypeVar, override

from oppertune.core.types import Context, PredictResponse
from oppertune.core.values import Categorical, CategoricalDict, Integer, IntegerDict, Real, RealDict, to_value

from ..base import Algorithm, _PredictResponse, _TuningRequest

__all__ = ("DDPG",)

_ParameterValueType = TypeVar("_ParameterValueType", bound=Union[str, int, float], default=Union[str, int, float])


class RewardData(TypedDict):
    next_state: Mapping[str, Any]


class DDPG(Algorithm[_ParameterValueType]):
    """The DDPG wrapper / helper class designed as an API for the DDPG agent."""

    class Meta:
        supported_parameter_types = (Categorical, Integer, Real)
        requires_untransformed_parameters = False
        supports_context = True
        supports_single_reward = False
        supports_sequence_of_rewards = True

    def __init__(
        self,
        parameters: Iterable[Union[Categorical, Integer, Real]],
        state_config: Iterable[Union[Categorical, Integer, Real, CategoricalDict, RealDict, IntegerDict]],
        epsilon: float = 0.1,
        epsilon_decay: float = 0.8,
        epsilon_decay_frequency: int = 10,
        hidden_width: int = 32,
        batch_size: int = 10,
        initial_random_steps: int = 10,
        gamma: float = 0.5,
        tau: float = 0.005,
        lr: float = 3e-4,
        learn_loop_count: int = 400,
        random_seed: Optional[int] = None,
    ):
        """DDPG wrapper has to be initialized mandatorily with parameters and state definition.

        Arguments:
            state_config: [Required] Tuple of state space feature definitions.
            parameters: [Required] Tuple of tunable parameter definitions.
            epsilon: [Optional] Fraction indicating trade-off in epsilon-greedy exploration
            epsilon_decay: [Optional] Decay factor for epsilon
            epsilon_decay_frequency: [Optional] Frequency or steps at which decay will be executed
            hidden_width: [Optional] Size of hidden layers of actor and critic networks
            batch_size: [Optional] Bactch size for training
            initial_random_steps: [Optional] Number of initial steps when agent is allowed to act randomly
            gamma: [Optional] Discount factor in episodic rewards
            tau: [Optional] Controls the variability of the critic networks
            lr: [Optional] Learning Rate
            learn_loop_count: [Optional] Numner of internal model update cycles between every set_reward
            random_seed: [Optional] Seeding for stability,
        """
        super().__init__(parameters, random_seed=random_seed)
        self.params: Tuple[Union[Categorical, Integer, Real], ...]  # For type hints
        self.state_config: Tuple[Union[Integer, Real, Categorical], ...] = tuple(to_value(sc) for sc in state_config)
        self.env = TuningEnvironment(
            state_config=self.state_config,
            parameters=self.params,
            random_seed=self._random_seed,
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
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_frequency = epsilon_decay_frequency
        self.replay_buffer = ReplayBuffer(self.env.state_dim, self.env.action_dim)
        self.learn_loop_count = learn_loop_count
        self.previous_reward = 0.0
        self.exploration_sampler = Bernoulli(1 - self.epsilon)

    @override
    def predict(
        self,
        context: Optional[Context] = None,
        predict_data: None = None,
    ) -> PredictResponse[_ParameterValueType]:
        return super().predict(context, predict_data)

    @override
    def _predict(self, context: Optional[Context] = None, predict_data: None = None) -> _PredictResponse:
        state: Union[Mapping[str, Any], None] = None if context is None else context.data
        if self._iteration < self.initial_random_steps:
            self.env.current_action = self.env.get_action_sample()  # a
            parameters = self.env.decode_actions(self.env.current_action)
            self._iteration += 1
            return _PredictResponse(parameters)

        current_state = self.env.current_state if state is None else self.env.encode_state(state)
        self.env.current_state = current_state

        action = self.agent.choose_action(current_state)
        if self._iteration % self.epsilon_decay_frequency == 0:
            self.epsilon *= self.epsilon_decay  # Decay the epsilon
            self.exploration_sampler.prob = 1 - self.epsilon

        should_explore = self.exploration_sampler.sample()
        self.env.current_action = action if should_explore else self.env.get_action_sample()

        parameters = self.env.decode_actions(self.env.current_action)
        return _PredictResponse(parameters)

    @override
    def _set_reward_for_sequence(
        self,
        tuning_requests: Sequence[_TuningRequest[_ParameterValueType, RewardData]],
    ) -> None:
        for i, tuning_request in enumerate(tuning_requests):
            cur_reward = tuning_request.reward
            cur_state = self.env.encode_state(tuning_request.context.data)
            cur_action = self.env.encode_action(tuning_request.prediction)
            next_state = None if tuning_request.reward_data is None else tuning_request.reward_data["next_state"]
            enc_next_state = cur_state if next_state is None else self.env.encode_state(next_state)
            done = torch.tensor(i == len(tuning_requests) - 1)
            if cur_reward is not None:
                self.replay_buffer.store(cur_state, cur_action, cur_reward, enc_next_state, done)

        self.agent.learn_wrapper(self.replay_buffer, self.learn_loop_count)

        self.env.current_state = enc_next_state
        self._iteration += 1

    @property
    @override
    def iteration(self) -> int:
        return self._iteration


class Actor(torch.nn.Module):
    """Actor network class.

    Directly predicts the policy State --> Action.
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
        a = self.max_action * self.sigmoid(self.l2(s))  # [-self.max_action, self.max_action]
        return a


class Critic(torch.nn.Module):
    """The Critic Network class.

    According to (state,action), directly calculate value function Q(state,action).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_width: int):
        super().__init__()
        self.l1 = torch.nn.Linear(state_dim + action_dim, hidden_width, dtype=torch.float64)
        self.l2 = torch.nn.Linear(hidden_width, 1, dtype=torch.float64)
        self.relu = torch.nn.ReLU()

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Concatenated state-action vector is pass through the citic network; Output is a scalar Q value."""
        sa = torch.cat([s, a], dim=1)
        q = self.relu(self.l1(sa))
        q = self.l2(q)
        return q


class ReplayBufferBatch(NamedTuple):
    """The named tuple representing a batch of transition samples from replay buffer.

    Elements:
        s: Current State in a transition.
        a: Current Action in a transition.
        r: Reward Signal obtained as feedback for action a in state s.
        s_: Next State due to action a.
        dw: Done flag highlighting if the the transition is the terminal step of the trajectory.
    """

    s: torch.Tensor
    a: torch.Tensor
    r: torch.Tensor
    s_: torch.Tensor
    dw: torch.Tensor


class ReplayBuffer:
    """Replay Buffer class for storing trajectories, to be used for training."""

    def __init__(self, state_dim: int, action_dim: int):
        """Replay buffer needs to be initialized with with state and action dimensions.

        Arguments:
            action_dim: Size of the action vector in internal representation
            state_dim: Size of the state vector in internal representation
        Properties:
            max_size: Total number of transitions it will store.
            count: The growing index of the transition record.
            size: Current size (#transitions) of the replay buffer.
            s: Current State in a transition.
            a: Current Action in a transition.
            r: Reward Signal obtained as feedback for action a in state s.
            s_: Next State due to action a.
            dw: Done flag highlighting if the the transition is the terminal step of the trajectory.
        """
        self.max_size = 1_000_000
        self.count = 0
        self.size = 0
        self.s = torch.zeros((self.max_size, state_dim), dtype=torch.float64)
        self.a = torch.zeros((self.max_size, action_dim), dtype=torch.float64)
        self.r = torch.zeros((self.max_size, 1), dtype=torch.float64)
        self.s_ = torch.zeros((self.max_size, state_dim), dtype=torch.float64)
        self.dw = torch.zeros((self.max_size, 1), dtype=torch.float64)

    def store(self, s: torch.Tensor, a: torch.Tensor, r: float, s_: torch.Tensor, dw: torch.Tensor) -> None:
        """The store method appends the buffer with new trajectories."""
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of transitions

    def sample(self, batch_size: int) -> ReplayBufferBatch:
        """The sample method samples a set of size = batch_size of trajectories from replay buffer."""
        i = torch.randperm(self.size)[:batch_size]  # Randomly sample an index
        return ReplayBufferBatch(self.s[i], self.a[i], self.r[i], self.s_[i], self.dw[i])


class Agent:
    """The DDPG Agent.

    The Agent class that constructs the RL agent using instances of the Actor and Critic classes
    as the actor-critic networks.
    This class also containts the learning methods that implements the DDPG updates

    Arguments:
        state_dim: [Required] integer value of the size of the state vector in the internal representation
        action_dim: [Required] integer value of the size of the action space vector in internal representation
    Other hyperparameters are documented inline
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
        """Helper function that executes the actor, flattens the predicted actions and detaches the gradients."""
        s = torch.unsqueeze(s, 0)
        return self.actor(s).detach().flatten()

    def learn(self, replay_buffer: ReplayBuffer) -> None:
        """The learn method contains the implementation of training updates to actor and critic networks.

        The training works with a sample batch of trajectories from replay buffer

        Arguments:
            replay_buffer: The active instance of the replay buffer
        """
        batch = replay_buffer.sample(self.batch_size)  # Sample a batch
        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            q = self.critic_target(batch.s_, self.actor_target(batch.s_))
            target_q = batch.r + self.gamma * (1 - batch.dw) * q

        # Compute the current Q and the critic loss
        cur_q = self.critic(batch.s, batch.a)
        critic_loss = self.mse_loss(target_q, cur_q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks
        self.critic.requires_grad_(False)

        # Compute the actor loss
        actor_loss = -self.critic(batch.s, self.actor(batch.s)).mean()

        # Optimize the actorstate
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        self.critic.requires_grad_(True)

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learn_wrapper(
        self,
        replay_buffer: ReplayBuffer,
        learn_loop_count: int = 100,
    ) -> None:
        """Wrapper function to call multiple model updates.

        Arguments:
            replay_buffer: Active instance of the replay buffer.
            learn_loop_count: #model update iterations.
        """
        for _ in range(learn_loop_count):
            self.learn(replay_buffer)


class TuningEnvironment(Env):
    """The Tuning Enironment Class.

    TuningEnvironment is the gymnasium based environment class.
    This is required for defining operable/learnable state and action spaces to be used in
    episodic RL algorithms such as DDPG.

    Arguments:
        1. state_config: [Required] Tuple of state feature definitions.
        2. parameters: [Required] Tuple of parameter definitions.
        3. random_seed: [Optional] Seed value for reproducibility.
    """

    def __init__(
        self,
        state_config: Sequence[Union[Categorical, Integer, Real]],
        parameters: Sequence[Union[Categorical, Integer, Real]],
        random_seed: Optional[int] = None,
    ):
        super().__init__()
        self.current_state: torch.Tensor = torch.empty()
        self.current_action: torch.Tensor = torch.empty()
        self.action_init: List[float] = []
        self.state_init: List[float] = []
        self.params = parameters
        self.state_config = state_config

        self.state_dim: int = 0
        self.action_dim: int = 0
        self.state_lb: List[float] = []
        self.state_ub: List[float] = []
        self.action_lb: List[float] = []
        self.action_ub: List[float] = []
        self.action_size: List[int] = []

        self._observation_space: Union[spaces.Box, None] = None
        self._action_space: Union[spaces.Box, None] = None
        self._action_param_mapping: Dict[str, Any] = {}
        self._state_feature_mapping: Dict[str, Any] = {}
        self._state_config: Dict[Any, Union[Categorical, Integer, Real]] = {}
        self._action_config: Dict[Any, Union[Categorical, Integer, Real]] = {}

        self.space_random_seed = random_seed
        self.parse_state_action_space()

    def parse_state_action_space(self) -> None:
        """Creates internal represation of state space and action space from state and parameter definitions."""
        self.state_dim = 0
        for state_feature in self.state_config:
            self._state_feature_mapping[state_feature.name] = self.state_dim
            if isinstance(state_feature, Categorical):
                category_embedding = [0.0] * state_feature.n_categories
                category_embedding[state_feature.encoded_value()] = 1.0
                self.state_init.extend(category_embedding)
                self._state_config[state_feature.name] = state_feature
                self.state_dim += state_feature.n_categories
            elif isinstance(state_feature, (Integer, Real)):
                _transformed_max = (
                    (state_feature.max - state_feature.min) // state_feature.step
                    if isinstance(state_feature, Integer)
                    else state_feature.max
                )
                self.state_init.append(
                    scale(
                        value=state_feature.val,
                        cur_min=state_feature.min,
                        cur_max=_transformed_max,
                        new_min=0.0,
                        new_max=1.0,
                    )
                )
                self._state_config[state_feature.name] = state_feature
                self.state_dim += 1
            else:
                raise TypeError(f"Invalid type for v: {type(state_feature)}")

        self.observation_space = spaces.Box(
            low=np.zeros(self.state_dim),
            high=np.ones(self.state_dim),
            shape=(self.state_dim,),
            dtype=np.float64,
        )

        self.action_param_mapping = {}
        self.action_dim = 0
        for param in self.params:
            self.action_param_mapping[param.name] = self.action_dim
            if isinstance(param, Categorical):
                category_embedding = [0.0] * param.n_categories
                category_embedding[param.encoded_value()] = 1.0
                self.action_init.extend(category_embedding)
                self.action_dim += param.n_categories
            elif isinstance(param, (Integer, Real)):
                self.action_init.append(
                    scale(value=param.val, cur_min=param.min, cur_max=param.max, new_min=0.0, new_max=1.0)
                )
                self._action_config[param.name] = param
                self.action_dim += 1
            else:
                raise TypeError(f"Invalid type for p: {type(param)}")

        self.action_space = spaces.Box(
            low=np.zeros(self.action_dim),
            high=np.ones(self.action_dim),
            shape=(self.action_dim,),
            dtype=np.float64,
        )

        if self.space_random_seed is not None:  # Seeding if available
            self.action_space.seed(self.space_random_seed)
            self.observation_space.seed(self.space_random_seed)

        self.max_action = torch.tensor(self.action_space.high.tolist(), dtype=torch.float64)
        self.current_state = torch.tensor(self.state_init, dtype=torch.float64)

    def decode_actions(self, action: Union[np.ndarray, torch.Tensor]) -> Dict[str, Union[str, int, float]]:
        """Transforms action predictions from internal representation to original parameter representation.

        Arguments:
        action: The action vector predicted by DDPG agent policy (actor) in internal representation
        """
        action_vec: List[float] = action.tolist()
        action_dict: Dict[str, Union[str, int, float]] = {}
        for param in self.params:
            if isinstance(param, Categorical):
                action_idx = self.action_param_mapping[param.name]
                category_embedding = action_vec[action_idx : action_idx + param.n_categories]
                hot_index = np.argmax(category_embedding)
                action_dict[param.name] = param.category(value=int(hot_index))
            elif isinstance(param, (Integer, Real)):
                action_idx = self.action_param_mapping[param.name]
                action_dict[param.name] = param.cast(
                    scale(
                        action_vec[action_idx],
                        cur_min=0.0,
                        cur_max=1.0,
                        new_min=param.min,
                        new_max=param.max,
                    )
                )
            else:
                raise TypeError(f"Invalid type for p: {type(param)}")

        return action_dict

    # Encode context or state
    def encode_state(self, state: Union[Mapping[str, Any], None]) -> torch.Tensor:
        """Transforms dictionary of state feature values from original space to internal representation.

        Arguments:
        state: Dictionary of state feature values passed as part of context
        """
        encoded_state = deepcopy(self.state_init)
        if state is None:
            return torch.tensor(encoded_state, dtype=torch.float64)

        for state_feature in self._state_config:
            state_feature_val = state[state_feature.name]
            if isinstance(state_feature, Categorical):
                position = self._state_feature_mapping[state_feature.name]
                category_embedding = [0.0] * state_feature.n_categories
                category_embedding[state_feature.encoded_value(category=state_feature_val)] = 1.0
                encoded_state[position : position + len(category_embedding)] = category_embedding
            elif isinstance(state_feature, (Integer, Real)):
                transformed_max = (
                    round((state_feature.max - state_feature.min) / state_feature.step)
                    if isinstance(state_feature, Integer)
                    else state_feature.max
                )
                position = self._state_feature_mapping[state_feature.name]
                encoded_state[position] = scale(
                    state_feature_val,
                    cur_min=state_feature.min,
                    cur_max=transformed_max,
                    new_min=0.0,
                    new_max=1.0,
                )
            else:
                raise TypeError(f"Invalid type for v: {type(state_feature)}")

        return torch.tensor(encoded_state, dtype=torch.float64)

    def encode_action(self, parameters: Mapping[str, _ParameterValueType]) -> torch.Tensor:
        """Transforms parameter values from original space to action vector in internal representation.

        Arguments:
            parameters: Dictionary of parameter values.
        """
        encoded_action = deepcopy(self.action_init)
        for param in self.params:
            parameter_value = parameters[param.name]
            if isinstance(param, Categorical):
                position = self.action_param_mapping[param.name]
                category_embedding = [0.0] * param.n_categories
                category_embedding[param.encoded_value(category=str(parameter_value))] = 1.0
                encoded_action[position : position + len(category_embedding)] = category_embedding
            elif isinstance(param, (Integer, Real)):
                position = self.action_param_mapping[param.name]
                encoded_action[position] = scale(
                    value=parameter_value,
                    cur_min=param.min,
                    cur_max=param.max,
                    new_min=0.0,
                    new_max=1.0,
                )
            else:
                raise TypeError(f"Invalid type for p: {type(param)}")

        return torch.tensor(encoded_action, dtype=torch.float64)

    def get_action_sample(self) -> torch.Tensor:
        """Samples a random action vector in internal representation."""
        return torch.tensor(self.action_space.sample())

    def get_action_init(self) -> torch.Tensor:
        """Fetches a initial action vector in internal representation."""
        return torch.tensor(self.action_init)


def scale(value: float, cur_min: float, cur_max: float, new_min: float, new_max: float) -> float:
    """Generic min-max scaling method required for encoding decoding state and parameter values.

    Argument:
        value: Input scalar value.
        cur_min: Current lower bound.
        cur_max: Current upper bound.
        new_min: New lower bound.
        new_max: New upper bound.
    """
    unit_value = (value - cur_min) / (cur_max - cur_min)
    new_value = new_min + unit_value * (new_max - new_min)
    return new_value


# Custom implementation of Bernoulli sampler required in exploration-explotation tradeoff
class Bernoulli:
    def __init__(self, prob: float = 0.5, random_seed: Optional[int] = None):
        """Bernoulli sampler.

        Argument:
            prob: Probability of getting True.
        """
        if prob < 0 or prob > 1:
            raise ValueError("prob must be between 0 and 1")

        self.prob = prob
        self._rng = Random(random_seed)

    def sample(self) -> bool:
        """Samples a boolean value with a probability <= 'prob'."""
        return self._rng.random() <= self.prob
