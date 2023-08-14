import numpy as np
import torch as th

from typing import Optional, Generator, Union, NamedTuple, Callable

from stable_baselines3.common.buffers import BaseBuffer
from gymnasium.spaces import Space, Discrete
from stable_baselines3.common.type_aliases import ReplayBufferSamples, RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class ActorCriticSamples(NamedTuple):
    observations: th.Tensor
    options_actions: th.Tensor
    returns: th.Tensor
    advantages: th.Tensor
    old_log_probs: th.Tensor
    old_values: th.Tensor
    next_values: th.Tensor


class TerminatorSamples(NamedTuple):
    observations: th.Tensor
    option_terminations: th.Tensor
    old_tn_log_probs: th.Tensor
    next_values: th.Tensor
    next_higher_level_value: th.Tensor


class OptionsRolloutBuffer(BaseBuffer):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    episode_starts: np.ndarray

    option_traces: np.ndarray
    option_terminations: np.ndarray
    option_tn_log_probs: np.ndarray

    # for both options and actions
    advantages: np.ndarray
    returns: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    next_values: np.ndarray

    def __init__(
            self,
            buffer_size: int,
            observation_space: Space,
            action_space: Space,
            option_hierarchy_size: int,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.option_hierarchy_size = option_hierarchy_size
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()
        self.total_transitions = self.buffer_size * self.n_envs

    def reset(self) -> None:
        base_shape = (self.buffer_size, self.n_envs)

        self.observations = np.zeros((*base_shape, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((*base_shape, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(base_shape, dtype=np.float32)
        self.episode_starts = np.zeros(base_shape, dtype=np.float32)

        self.option_traces = np.zeros((*base_shape, self.option_hierarchy_size), dtype=np.float32)
        self.option_terminations = np.zeros((*base_shape, self.option_hierarchy_size), dtype=np.float32)
        self.option_tn_log_probs = np.zeros((*base_shape, self.option_hierarchy_size), dtype=np.float32)

        self.advantages = np.zeros((*base_shape, self.option_hierarchy_size + 1), dtype=np.float32)
        self.advantages[:] = np.nan
        self.returns = np.zeros((*base_shape, self.option_hierarchy_size + 1), dtype=np.float32)
        self.log_probs = np.zeros((*base_shape, self.option_hierarchy_size + 1), dtype=np.float32)
        self.values = np.zeros((*base_shape, self.option_hierarchy_size + 1), dtype=np.float32)
        # Keeps value of next state as determined by the option saved in option_traces (not the new option)
        self.next_values = np.zeros((*base_shape, self.option_hierarchy_size + 1), dtype=np.float32)

        self.generator_ready = False
        super().reset()

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            next_value: th.Tensor,
            log_probs: th.Tensor,
            option_trace: th.Tensor,
            termination: th.Tensor,
            termination_log_prob: th.Tensor,
    ) -> None:
        # Preparations (as done in SB3 RolloutBuffer)
        if len(log_probs.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_probs = log_probs.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()

        self.option_traces[self.pos] = np.array(option_trace).copy()
        self.log_probs[self.pos] = log_probs.clone().cpu().numpy()
        self.values[self.pos] = value.clone().cpu().numpy()
        self.next_values[self.pos] = next_value.clone().cpu().numpy()

        self.option_terminations[self.pos] = termination.clone().cpu().numpy()
        self.option_tn_log_probs[self.pos] = termination_log_prob.clone().cpu().numpy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[ActorCriticSamples, None, None]:
        raise NotImplementedError

    def get_actor_critic_train_data(
            self,
            level: int = None,
            option_id: int = None,
            batch_size: Optional[int] = None
    ) -> Generator[ActorCriticSamples, None, None]:
        assert self.full, "Buffer is not full"

        if not self.generator_ready:
            self._prepare_generator()

        is_global_policy = level is None or option_id is None

        if is_global_policy:
            decisions = self.get_global_policy_decisions()
            level = -1
        else:
            decisions = self.get_option_decisions(level, option_id)

        indices = np.where(decisions)[0]

        return self._get(indices, level, self._get_actor_critic_samples, batch_size)

    def get_terminator_train_data(
            self,
            level: int,
            option_id: int,
            batch_size: Optional[int] = None
    ) -> Generator[TerminatorSamples, None, None]:
        assert self.full, "Buffer is not full"

        if not self.generator_ready:
            self._prepare_generator()

        option_active = self.get_option_active(level, option_id)
        indices = np.where(option_active)[0]

        return self._get(indices, level, self._get_terminator_samples, batch_size)

    def _get(
            self,
            indices: np.array,
            level: int,
            samples_getter: Callable,
            batch_size: Optional[int] = None,
    ) -> Generator:
        n_transitions = len(indices)
        if n_transitions == 0:
            return

        # Randomize index order
        np.random.shuffle(indices)

        if batch_size is None:
            # Return everything, don't create mini-batches
            batch_size = n_transitions

        start_idx = 0
        while start_idx < n_transitions:
            yield samples_getter(indices[start_idx: start_idx + batch_size], level)
            start_idx += batch_size

    def _prepare_generator(self):
        _tensor_names = [
            "observations",
            "actions",
            "rewards",
            "episode_starts",

            "option_traces",
            "option_terminations",
            "option_tn_log_probs",

            "advantages",
            "returns",
            "log_probs",
            "values",
            "next_values",
        ]

        for tensor in _tensor_names:
            self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
        self.generator_ready = True

    def _get_samples(
            self,
            batch_inds: np.ndarray,
            env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        raise NotImplementedError

    def _get_actor_critic_samples(
            self,
            indices: np.ndarray,
            level: int = -1,
    ) -> ActorCriticSamples:
        if level == self.option_hierarchy_size - 1:
            # Lowest-level options decide over actions
            options_actions = self.actions[indices]
        else:
            # Higher-level options decide over options
            options_actions = self.option_traces[indices, ..., level + 1]

        data = (
            self.observations[indices],
            options_actions,
            self.returns[indices, ..., level + 1],
            self.advantages[indices, ..., level + 1],
            self.log_probs[indices, ..., level + 1],
            self.values[indices, ..., level + 1],
            self.next_values[indices, ..., level + 1],
        )
        return ActorCriticSamples(*tuple(map(self.to_torch, data)))

    def _get_terminator_samples(
            self,
            indices: np.ndarray,
            level: int,
    ) -> TerminatorSamples:
        data = (
            self.observations[indices],
            self.option_terminations[indices, level],
            self.option_tn_log_probs[indices, level],
            self.next_values[indices, ..., level + 1],
            self.next_values[indices, ..., level]
        )
        return TerminatorSamples(*tuple(map(self.to_torch, data)))

    def get_option_active(
            self,
            level: Union[int, th.Tensor],
            option_id: Union[int, th.Tensor]
    ) -> th.Tensor:
        """Returns the indices for all transitions where the specified option was active,
        i.e., is contained in an option trace."""
        assert self.generator_ready
        return self.option_traces[:, level] == option_id

    def get_option_starts(self, level: Union[int, th.Tensor] = None):
        if level is None or self.option_hierarchy_size == 0:
            shape = self.option_terminations.shape
            level = slice(None)
        else:
            shape = self.option_terminations.shape[:-1]
        option_starts = np.ones(shape, dtype='bool')
        option_starts[1:] = self.option_terminations[:-1, ..., level]
        return option_starts

    def get_global_policy_decisions(self) -> np.array:
        if self.option_hierarchy_size > 0:
            return self.get_option_starts(0)
        else:
            return np.ones(self.values.shape, dtype='bool')

    def get_option_decisions(self, level: Union[int, th.Tensor], option_id: Union[int, th.Tensor]) -> np.array:
        option_decisions = self.get_option_active(level, option_id)
        if level + 1 < self.option_hierarchy_size:
            lower_level_starts = self.get_option_starts(level + 1)
            option_decisions &= lower_level_starts
        return option_decisions

    def get_values_upon_arrival(self):
        """Returns for each option (not for the global policy) the value upon
        arrival. Implements the function U_omega(s')."""
        termination_prob = np.exp(self.option_tn_log_probs)
        next_outer_value = self.next_values[..., :-1]  # in context of higher-level option
        next_local_value = self.next_values[..., 1:]  # in context of this option
        return termination_prob * next_outer_value + (1 - termination_prob) * next_local_value

    def compute_returns_and_advantage(self, dones: np.ndarray) -> None:
        # Identify policy decision points
        decisions = np.ones(self.advantages.shape, dtype='bool')
        decisions[..., :-1] = self.get_option_starts()  # Lowest-level option always makes a decision

        # Keep track of properties between decision points
        reward = np.zeros(self.advantages.shape[1:], dtype=np.float32)
        next_decision = np.ones(self.advantages.shape[1:], dtype=bool)
        next_value = np.zeros(self.advantages.shape[1:], dtype=np.float32)
        last_gae = np.zeros(self.advantages.shape[1:], dtype=np.float32)

        values_upon_arrival = self.get_values_upon_arrival()

        policy_continues = np.ones(self.advantages.shape[1:], dtype=bool)

        episode_continues = ~dones.astype(bool)

        for step in reversed(range(self.buffer_size)):
            # Update next value for policies before their next decision
            next_value_tmp = self.next_values[step]
            next_value_tmp[..., 1:] = values_upon_arrival[step]
            next_value[next_decision] = next_value_tmp[next_decision]

            # Only policies that actively made a decision receive an advantage estimate
            decision = decisions[step]

            # Determine policy continuation/termination
            episode_continues = np.expand_dims(episode_continues, axis=1)
            option_terminates = self.option_terminations[step].astype(bool)
            # if step < self.buffer_size - 1:  # advantage carryover
            #     option_terminates |= self.option_traces[step] == self.option_traces[step + 1]
            policy_continues[..., 0] = True  # global policy always continues
            policy_continues[..., 1:] = ~option_terminates  # options continue if they don't terminate
            policy_continues &= episode_continues

            reward += np.expand_dims(self.rewards[step], axis=1)

            # Compute Generalized Advantage Estimate (GAE)
            td_0_estimate = reward + episode_continues * self.gamma * next_value
            delta = td_0_estimate - self.values[step]  # TD error
            gae = delta + policy_continues * self.gamma * self.gae_lambda * last_gae
            self.advantages[step, decision] = gae[decision]
            last_gae[decision] = gae[decision]

            reward[decision] = 0  # also takes done episodes into account, implicitly

            episode_continues = ~self.episode_starts[step].astype(bool)
            next_decision = decision

        # TD(lambda) estimator, see GitHub PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values
