import numpy as np
import torch as th

from typing import Optional, Generator

from stable_baselines3.common.buffers import RolloutBuffer, RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class OptionsRolloutBufferSamples(RolloutBufferSamples):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_pi_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    options: th.Tensor
    terminations: th.Tensor
    old_tn_log_prob: th.Tensor


class OptionsRolloutBuffer(RolloutBuffer):
    options: np.ndarray
    terminations: np.ndarray
    termination_log_probs: np.ndarray

    def reset(self) -> None:
        self.options = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.terminations = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.termination_log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    # Method signature extended by Liskov substitution principle
    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
            option: th.Tensor,
            terminations: th.Tensor,
            termination_log_probs: th.Tensor,
    ) -> None:
        self.options[self.pos] = option.clone().cpu().numpy()
        self.terminations[self.pos] = terminations.clone().cpu().numpy()
        self.termination_log_probs[self.pos] = termination_log_probs.clone().cpu().numpy()
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: Optional[int] = None) -> Generator[OptionsRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "options",
                "terminations",
                "termination_log_probs",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> OptionsRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.options[batch_inds].flatten(),
            self.terminations[batch_inds].flatten(),
            self.termination_log_probs[batch_inds].flatten(),
        )
        return OptionsRolloutBufferSamples(*tuple(map(self.to_torch, data)))
