from stable_baselines3.common.monitor import Monitor
from typing import Any, SupportsFloat
import time
import numpy as np

class TrainMonitor(Monitor):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    :param override_existing: appends to file if ``filename`` exists, otherwise
        override existing files (default)
    """

    def __init__(
        self,
        env,
        filename = None,
        allow_early_resets = True,
        reset_keywords = (),
        info_keywords = (),
        override_existing = True,
    ):
        super().__init__(env, filename, allow_early_resets, reset_keywords, info_keywords, override_existing)
        self.all_rewards = []
        self.episode_all_returns = []


    def reset(self, **kwargs): 
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        self.all_rewards = []
        return super().reset(**kwargs)

    def step(self, action): 
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "all_rewards" in info:
            self.all_rewards.append(info["all_rewards"])

        self.rewards.append(float(reward))

        if terminated or truncated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            # shape: (n_steps, n_rewards)
            all_rewards = np.array(self.all_rewards)
            a_r = all_rewards.sum(axis=0)
            self.episode_all_returns.append(a_r.tolist())

            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            # ep_info["all_rewards"] = a_r.tolist()
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info