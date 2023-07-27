import numpy as np
import torch as th

from stable_baselines3.common.buffers import RolloutBuffer


class OptionsRolloutBuffer(RolloutBuffer):
    options: np.ndarray
    option_terminations: np.ndarray

    def reset(self) -> None:
        self.options = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.option_terminations = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
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
            option_termination: th.Tensor,
            termination_log_probs: th.Tensor,
    ) -> None:
        self.options[self.pos] = option.clone().cpu().numpy()
        self.option_terminations[self.pos] = option_termination.clone().cpu().numpy()
        self.termination_log_probs[self.pos] = termination_log_probs.clone().cpu().numpy()
        super().add(obs, action, reward, episode_start, value, log_prob)
