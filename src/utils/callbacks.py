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

from options.ppo import OptionsPPO
from utils.console import green


class OptionEvalCallback(EvalCallback):
    model: OptionsPPO

    def __init__(self, *args, max_episode_len: int = None, early_stop_on_no_reward: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_episode_len = max_episode_len
        self.early_stop_on_no_reward = early_stop_on_no_reward

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

            # if self.verbose >= 1:
            #     print("\nRunning evaluation... ", end="")

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                max_episode_len=self.max_episode_len,
                early_stop_on_no_reward=self.early_stop_on_no_reward,
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

            ret, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = ret

            if self.verbose >= 1:
                # Override the two progress bars by moving the cursor to beginning of previous line
                print(f"\033[F\r")
                ret_text = f"{ret:.2f}"
                if ret > self.best_mean_reward:
                    ret_text = green(ret_text)
                print(f"Ret: {ret_text} +/- {std_reward:.2f} - "
                      f"Len: {ep_length:.2f} +/- {std_ep_length:.2f}\033[K")

            # Add to current Logger
            self.logger.record("eval/return", float(ret))
            self.logger.record("eval/episode_length", ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if ret > self.best_mean_reward:
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = ret
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
    Custom callback for plotting the original return (useful in cases
    where reward shaping is applied).
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
        self.logger.record("rollout/original_return", np.mean(list(self.buffer)))


class SaveBestModelCallback(BaseCallback):
    """Saves both the model and the model's VecNormalize env's statistics."""

    def __init__(self, save_path: str):
        super(SaveBestModelCallback, self).__init__()
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        save_path_name = os.path.join(self.save_path, "best_vecnormalize.pkl")
        vec_norm_env = self.model.get_vec_normalize_env()
        if vec_norm_env is not None:
            vec_norm_env.save(save_path_name)

        self.model.save(os.path.join(self.save_path, "best_model"))

        return True


def evaluate_policy(
        model: OptionsPPO,
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        max_episode_len: int = None,
        early_stop_on_no_reward: int = None,
        warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """Similar to evaluate_policy from stable_baselines3.common.evaluation
    but for options.
    :param model:
    :param env:
    :param n_eval_episodes:
    :param deterministic:
    :param render:
    :param callback:
    :param reward_threshold:
    :param return_episode_rewards:
    :param max_episode_len:
    :param early_stop_on_no_reward: Premature stop of episode if last early_stop transitions
        had no positive reward.
    :param warn:
    :return: """

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
    last_positive_rewards = np.zeros(n_envs)
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    option_terminations = th.ones(env.num_envs, model.hierarchy_size, device=model.device, dtype=th.bool)
    options = th.zeros(env.num_envs, model.hierarchy_size, device=model.device, dtype=th.long)

    while (episode_counts < episode_count_targets).any():
        (options, actions), _, _ = model.forward_all(observations, options, option_terminations, deterministic)

        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1

        last_positive_rewards[rewards > 0] = 0
        last_positive_rewards[rewards <= 0] += 1

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if (early_stop_on_no_reward is not None and last_positive_rewards[i] >= early_stop_on_no_reward
                        or max_episode_len is not None and current_lengths[i] >= max_episode_len):
                    dones[i] = True
                    stop_early = True
                    new_observations[i] = env.env_method("reset", indices=[i])[0][0]
                else:
                    stop_early = False

                if dones[i]:
                    if is_monitor_wrapped and not stop_early:
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
                    last_positive_rewards[i] = 0

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
                   eval_kwargs: dict) -> CallbackList:
    checkpoint_frequency = 1_000_000
    rtpt_frequency = 100_000

    rtpt_iters = total_timestamps // rtpt_frequency
    eval_callback = OptionEvalCallback(
        eval_env,
        n_eval_episodes=n_eval_episodes,
        callback_on_new_best=SaveBestModelCallback(str(ckpt_path)),
        log_path=str(ckpt_path),
        eval_freq=max(eval_kwargs.pop("frequency") // n_envs, 1),
        **eval_kwargs)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(checkpoint_frequency // n_envs, 1),
        save_path=str(ckpt_path),
        name_prefix="model",
        save_vecnormalize=True)

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
