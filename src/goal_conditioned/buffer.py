import numpy as np

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

# simply allow to store multiple rewards for the same transition
class MultiRewardReplayBuffer(ReplayBuffer):
    def __init__(self, n_rewards, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_rewards = n_rewards
        self.rewards = np.zeros((self.buffer_size, self.n_envs, n_rewards)) 
    
    def _get_samples(self, batch_inds: np.ndarray, env = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1, self.n_rewards), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))