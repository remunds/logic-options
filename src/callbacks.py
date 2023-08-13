import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Union
from typing import Tuple
from pathlib import Path
from collections import deque

import gymnasium as gym
import numpy as np
import torch as th
from rtpt import RTPT
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.vec_env import VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, CheckpointCallback, \
    EveryNTimesteps

from options_ppo import OptionsPPO


class OptionEvalCallback(EvalCallback):
    model: OptionsPPO

    def init_callback(self, model: OptionsPPO) -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        super().init_callback(model)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0 or self.n_calls == 1:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Step {self.num_timesteps} - Eval results:")
                print(f"\tReturn: \t{mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"\tEpisode length: \t{mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))  # TODO: rename
            self.logger.record("eval/mean_ep_length", mean_ep_length)  # TODO: rename

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


class RtptCallback(BaseCallback):
    def __init__(self, exp_name, max_iter, verbose=0):
        super(RtptCallback, self).__init__(verbose)
        self.rtpt = RTPT(name_initials="QD",
                         experiment_name=exp_name,
                         max_iterations=max_iter)
        self.rtpt.start()

    def _on_step(self) -> bool:
        self.rtpt.step()
        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, n_envs, verbose=0):
        self.n_envs = n_envs
        self.buffer = deque(maxlen=100)  # ppo default stat window
        super().__init__(verbose)

    def _on_step(self) -> None:
        ep_rewards = self.training_env.get_attr("ep_env_reward", range(self.n_envs))
        for rew in ep_rewards:
            if rew is not None:
                self.buffer.extend([rew])

    def on_rollout_end(self) -> None:
        buff_list = list(self.buffer)
        if len(buff_list) == 0:
            return
        self.logger.record("rollout/ep_env_rew_mean", np.mean(list(self.buffer)))


def evaluate_policy(
        model: OptionsPPO,
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """Same as evaluate_policy from stable_baselines3.common.evaluation
    but for options."""
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    option_terminations = th.ones(env.num_envs, model.hierarchy_size).type(th.BoolTensor)
    options = th.zeros(env.num_envs, model.hierarchy_size).type(th.LongTensor)

    while (episode_counts < episode_count_targets).any():
        (options, actions), _, _ = model.forward_all(observations, options, option_terminations, deterministic)

        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        option_terminations, _ = model.forward_all_terminators(new_observations, options)
        option_terminations[dones] = True

        observations = new_observations

        if render:
            env.render()

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


def init_callbacks(exp_name: str,
                   total_timestamps: int,
                   object_centric: bool,
                   n_envs: int,
                   eval_env,
                   n_eval_episodes: int,
                   ckpt_path: Path,
                   eval_frequency: int = 100_000,
                   eval_callback_cls=OptionEvalCallback,
                   eval_render: bool = False,
                   eval_deterministic: bool = True) -> CallbackList:
    checkpoint_frequency = 1_000_000
    rtpt_frequency = 100_000

    rtpt_iters = total_timestamps // rtpt_frequency
    eval_callback = eval_callback_cls(
        eval_env,
        n_eval_episodes=n_eval_episodes,
        best_model_save_path=str(ckpt_path),
        log_path=str(ckpt_path),
        eval_freq=max(eval_frequency // n_envs, 1),
        deterministic=eval_deterministic,
        render=eval_render)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(checkpoint_frequency // n_envs, 1),
        save_path=str(ckpt_path),
        name_prefix="model",
        save_vecnormalize=False)

    rtpt_callback = RtptCallback(
        exp_name=exp_name,
        max_iter=rtpt_iters)

    n_callback = EveryNTimesteps(
        n_steps=rtpt_frequency,
        callback=rtpt_callback)

    callbacks = [checkpoint_callback, eval_callback, n_callback]

    if object_centric:
        callbacks.append(TensorboardCallback(n_envs=n_envs))

    return CallbackList(callbacks)
