import numpy as np
import torch as th

from typing import Optional, Generator, Union, NamedTuple

from stable_baselines3.common.buffers import BaseBuffer, RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium.spaces import Space, Discrete


class OptionsRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    returns: th.Tensor

    option_traces: th.Tensor
    advantages: th.Tensor
    old_log_probs: th.Tensor
    old_values: th.Tensor
    next_values: th.Tensor

    option_terminations: th.Tensor
    old_tn_log_prob: th.Tensor


class OptionsRolloutBuffer(BaseBuffer):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray

    option_traces: np.ndarray
    advantages: np.ndarray  # for both options and actions
    log_probs: np.ndarray  # for both options and actions
    values: np.ndarray  # for both options and actions
    next_values: np.ndarray  # for both options and actions

    option_terminations: np.ndarray
    option_tn_log_probs: np.ndarray

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

    def reset(self) -> None:
        base_shape = (self.buffer_size, self.n_envs)

        self.observations = np.zeros((*base_shape, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((*base_shape, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(base_shape, dtype=np.float32)
        self.returns = np.zeros((*base_shape, self.option_hierarchy_size + 1), dtype=np.float32)
        self.episode_starts = np.zeros(base_shape, dtype=np.float32)

        self.option_traces = np.zeros((*base_shape, self.option_hierarchy_size), dtype=np.int32)
        self.advantages = np.zeros((*base_shape, self.option_hierarchy_size + 1), dtype=np.float32)
        self.log_probs = np.zeros((*base_shape, self.option_hierarchy_size + 1), dtype=np.float32)
        self.values = np.zeros((*base_shape, self.option_hierarchy_size + 1), dtype=np.float32)
        # Keeps value of next state as determined by the option saved in option_traces (not the new option)
        self.next_values = np.zeros((*base_shape, self.option_hierarchy_size + 1), dtype=np.float32)

        self.option_terminations = np.zeros((*base_shape, self.option_hierarchy_size), dtype=bool)
        self.option_tn_log_probs = np.zeros((*base_shape, self.option_hierarchy_size), dtype=np.float32)

        super().reset()

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            values: th.Tensor,
            next_values: th.Tensor,
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
        self.values[self.pos] = values.clone().cpu().numpy()
        self.next_values[self.pos] = next_values.clone().cpu().numpy()

        self.option_terminations[self.pos] = termination.clone().cpu().numpy()
        self.option_tn_log_probs[self.pos] = termination_log_prob.clone().cpu().numpy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[OptionsRolloutBufferSamples, None, None]:
        """Returns data that is used to train the global (inter-option) policy."""
        assert self.full, "Buffer is not full"

        if not self.generator_ready:
            self._prepare_generator()

        decisions = self.get_global_policy_decisions()
        indices = np.where(decisions)[0]

        yield self._get(indices, batch_size)

    def get_for_option(
            self,
            level: Union[int, th.Tensor],
            option_id: Union[int, th.Tensor],
            batch_size: Optional[int] = None
    ) -> Generator[OptionsRolloutBufferSamples, None, None]:
        """Returns data that is used to train the specified option."""
        assert self.full, "Buffer is not full"

        if not self.generator_ready:
            self._prepare_generator()

        option_decisions = self.get_option_decisions(level, option_id)
        indices = np.where(option_decisions)[0]

        yield self._get(indices, batch_size)

    def _get(
            self,
            indices: np.array,
            batch_size: Optional[int] = None
    ) -> Generator[OptionsRolloutBufferSamples, None, None]:
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
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _prepare_generator(self):
        _tensor_names = [
            "observations",
            "actions",
            "returns",

            "option_traces",
            "advantages",
            "log_probs",
            "values",
            "next_values",

            "option_terminations",
            "option_tn_log_probs",
        ]

        for tensor in _tensor_names:
            self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
        self.generator_ready = True

    def _get_samples(
            self,
            indices: np.ndarray,
            env: Optional[VecNormalize] = None,
    ) -> OptionsRolloutBufferSamples:
        data = (
            self.observations[indices],
            self.actions[indices],
            self.returns[indices],

            self.option_traces[indices],
            self.advantages[indices],
            self.log_probs[indices],
            self.values[indices],
            self.next_values[indices],

            self.option_terminations[indices],
            self.option_tn_log_probs[indices],
        )
        return OptionsRolloutBufferSamples(*tuple(map(self.to_torch, data)))

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
        if level is None:
            shape = self.rewards.shape
            level = Ellipsis
        else:
            shape = self.option_terminations.shape
        option_starts = np.ones(shape, dtype='bool')
        option_starts[1:] = self.option_terminations[:-1, ..., level]
        return option_starts

    def get_global_policy_decisions(self) -> np.array:
        return self.get_option_starts(0)

    def get_option_decisions(self, level: Union[int, th.Tensor], option_id: Union[int, th.Tensor]) -> np.array:
        option_decisions = self.get_option_active(level, option_id)
        if level + 1 < self.option_hierarchy_size:
            lower_level_starts = self.get_option_starts(level + 1)
            option_decisions &= lower_level_starts
        return option_decisions

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy()

        termination_probs = np.exp(self.option_tn_log_probs)

        self.advantages[:] = np.nan
        last_gae = np.zeros(self.advantages.shape[1:])
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]

            next_outer_value = next_values[..., :-1]  # in context of higher-level option
            next_local_value = self.next_values[step, ..., 1:]  # in context of this option
            termination_prob = termination_probs[step]
            next_values[..., 1:] = termination_prob * next_outer_value + (1 - termination_prob) * next_local_value

            option_starts = np.ones((self.advantages.shape[1:]), dtype='bool')
            if step > 0:
                option_starts[:, :-1] = self.option_terminations[step - 1]  # Actions always "start" anew

            # Adjust shapes
            rewards = np.expand_dims(self.rewards[step], axis=1)
            next_non_terminal = np.expand_dims(next_non_terminal, axis=1)

            td_0_estimate = rewards + self.gamma * next_values * next_non_terminal
            delta = td_0_estimate - self.values[step]  # TD error
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step, option_starts] = gae[option_starts]
            last_gae[option_starts] = gae[option_starts]

        # TD(lambda) estimator, see GitHub PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values
