from __future__ import annotations

import sys
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import torch as th
import yaml
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import obs_as_tensor, explained_variance, update_learning_rate, get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo.ppo import PPO
from torch.nn.functional import mse_loss
from tqdm import tqdm

from envs.common import init_vec_env
from envs.util import get_env_identifier
from options.agent import OptionsAgent
from options.option import Terminator
from options.rollout_buffer import OptionsRolloutBuffer
from utils.common import get_option_name, get_most_recent_checkpoint_steps


class OptionsPPO(PPO):
    """
    Proximal Policy Optimization (PPO) algorithm with options

    :param options_hierarchy: Number of options
    :param net_arch: The network architecture of each individual option policy,
        specified as a sequence of dense layer widths
    :param options_termination_reg: Variable xi. If increased, option termination
            probability gets decreased, i.e., longer options are encouraged. Set
            to 0 do disable this regularization.
    :param **kwargs: PPO keyworded arguments, see SB3 docs:
        https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    """

    rollout_buffer: OptionsRolloutBuffer
    policy: OptionsAgent
    progress_total: tqdm
    progress_rollout_train: tqdm
    _last_active_options: np.array
    _last_option_terminations: np.array

    def __init__(self,
                 meta_learning_rate: Union[float, Schedule] = 3e-4,
                 meta_policy_ent_coef: float = 0,
                 meta_policy_clip_range: Union[float, Schedule] = 1,
                 meta_value_fn_coef: float = 1,
                 meta_value_fn_clip_range: Union[float, Schedule] = None,
                 options_learning_rate: Union[float, Schedule] = 3e-4,
                 options_policy_ent_coef: float = 0,
                 options_policy_clip_range: Union[float, Schedule] = None,
                 options_value_fn_coef: float = 1,
                 options_value_fn_clip_range: Union[float, Schedule] = None,
                 options_terminator_ent_coef: float = 0,
                 options_terminator_clip_range: Union[float, Schedule] = None,
                 options_termination_reg: float = 0,
                 **kwargs):
        kwargs["policy"] = OptionsAgent
        super().__init__(**kwargs)

        self.meta_learning_rate = meta_learning_rate
        self.meta_pi_ent_coef = meta_policy_ent_coef
        self.meta_pi_clip_range = meta_policy_clip_range
        self.meta_vf_coef = meta_value_fn_coef
        self.meta_vf_clip_range = meta_value_fn_clip_range

        self.options_learning_rate = options_learning_rate
        self.options_pi_ent_coef = options_policy_ent_coef
        self.options_pi_clip_range = options_policy_clip_range
        self.options_vf_coef = options_value_fn_coef
        self.options_vf_clip_range = options_value_fn_clip_range
        self.options_tn_ent_coef = options_terminator_ent_coef
        self.options_tn_clip_range = options_terminator_clip_range
        self.options_tn_reg = options_termination_reg

        self._setup_clip_ranges()

    def _setup_model(self) -> None:
        super()._setup_model()

        self.hierarchy_size = self.policy.hierarchy_size

        self._last_option_terminations = th.ones(self.env.num_envs,
                                                 self.policy.hierarchy_size,
                                                 device=self.device,
                                                 dtype=th.bool)
        self._last_active_options = th.zeros(self.env.num_envs,
                                             self.policy.hierarchy_size,
                                             device=self.device,
                                             dtype=th.long)

        self.rollout_buffer = OptionsRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            option_hierarchy_size=self.hierarchy_size,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            seed=self.seed,
        )

    def _setup_clip_ranges(self) -> None:
        """Initializes all clip range schedule functions."""
        self.meta_pi_clip_range = get_schedule_fn(self.meta_pi_clip_range)
        if self.meta_vf_clip_range is not None:
            self.meta_vf_clip_range = get_schedule_fn(self.meta_vf_clip_range)
        if self.options_pi_clip_range is not None:
            self.options_pi_clip_range = get_schedule_fn(self.options_pi_clip_range)
        if self.options_tn_clip_range is not None:
            self.options_tn_clip_range = get_schedule_fn(self.options_tn_clip_range)
        if self.options_vf_clip_range is not None:
            self.options_vf_clip_range = get_schedule_fn(self.options_vf_clip_range)

    def learn(self, total_timesteps: int, **kwargs):
        self.progress_total = tqdm(total=total_timesteps, file=sys.stdout, desc="Total steps",
                                   position=0, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',
                                   leave=False, unit_scale=True)
        super().learn(total_timesteps, **kwargs)

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: OptionsRolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        dones = None

        self.progress_rollout_train = tqdm(total=n_rollout_steps * self.n_envs,
                                           file=sys.stdout, desc="Collecting rollout",
                                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',
                                           position=1,
                                           leave=False)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                (options, actions), values, log_probs = \
                    self.policy.forward_all(obs_tensor, self._last_active_options, self._last_option_terminations)

            actions = actions.cpu().numpy().astype(self.action_space.dtype)

            # Clip the actions to avoid out of bound error
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # Perform actions
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs
            self.progress_total.update(env.num_envs)

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.meta_policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.meta_policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            # When an episode terminates, new_obs is 1st frame of new episode, so we need
            # to replace it and use the most recent obs as a surrogate for termination and
            # value prediction
            next_obs = np.array(new_obs).copy()
            next_obs[dones] = self._last_obs[dones]

            with th.no_grad():
                next_obs_tensor = obs_as_tensor(next_obs, self.device)
                terminations, _ = self.policy.forward_all_terminators(next_obs_tensor, options)

                # If a higher-level option exits, all lower-level options exit, too
                for level in range(1, self.policy.hierarchy_size):
                    terminations[:, level] |= terminations[:, level - 1]

                # Compute log probability for terminations (note: not continuations)
                true_tensor = th.ones(terminations.shape, device=self.device)
                tn_log_probs = self.policy.evaluate_terminations(next_obs_tensor, options, true_tensor)

                assert not th.any(th.isinf(tn_log_probs))

                # TODO: efficiency can be improved as next values are needed only for terminated options
                next_values = self.policy.predict_all_values(next_obs_tensor, options)

            # Save observed data to rollout buffer
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                dones,
                self._last_episode_starts,
                next_obs,
                values,
                next_values,
                log_probs,
                options.cpu(),
                terminations,
                tn_log_probs,
            )

            # Prepare next iteration
            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_active_options = options
            terminations[dones] = True
            self._last_option_terminations = terminations

            self.progress_rollout_train.update(self.n_envs)

        progress_percent = (1 - self._current_progress_remaining) * 100

        # print(f"\rTotal steps: {num2text(self.num_timesteps)} ({progress_percent:.1f} %) - "
        #       f"Total updates: {num2text(self._n_updates)}")

        rollout_buffer.compute_returns_and_advantage(dones=dones)

        # Log option activity
        for level, n_options in enumerate(self.policy.hierarchy_shape):
            level_options = self.rollout_buffer.option_traces[:, :, level].astype(int)
            level_terminations = self.rollout_buffer.option_terminations[:, :, level].astype(bool)
            dones = self.rollout_buffer.dones.astype(bool)

            for position in range(n_options):
                option_active = level_options == position
                option_count = np.sum(option_active)
                option_activity_share = option_count / self.rollout_buffer.total_transitions

                if option_count > 0:
                    option_terminates = level_terminations[option_active] | dones[option_active]
                    option_length = option_count / (np.sum(option_terminates) + 1)
                    # FIXME: options can be longer than a rollout, need to track this
                else:
                    option_length = np.nan

                option_name = get_option_name(level, position)
                self.logger.record(option_name + "/activity_share", option_activity_share)
                self.logger.record(option_name + "/length", option_length)

        self.progress_rollout_train.close()

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        self.progress_rollout_train = tqdm(total=int(self.policy.n_policies * self.n_epochs),
                                           file=sys.stdout, desc="Training",
                                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',
                                           position=1,
                                           leave=False)

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Train meta policy
        self._train_actor_critic()

        # Train actor, critic, and terminator of each option
        for level, n_options in enumerate(self.policy.hierarchy_shape):
            for position in range(n_options):
                self._train_actor_critic(level, position)
                self._train_terminator(level, position)

        self._n_updates += self.n_epochs

        # Parameter tracking
        clip_range, clip_range_vf = self._get_current_meta_clip_ranges()
        self.logger.record("hyperparameter_schedule/meta_pi_clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("hyperparameter_schedule/meta_vf_clip_range", clip_range_vf)
        if self.hierarchy_size > 0:
            clip_range, clip_range_vf = self._get_current_options_clip_ranges()
            if self.clip_range is not None:
                self.logger.record("hyperparameter_schedule/options_pi_clip_range", clip_range)
            if self.clip_range_vf is not None:
                self.logger.record("hyperparameter_schedule/options_vf_clip_range", clip_range_vf)
        self.logger.record("n_updates", self._n_updates, exclude="tensorboard")

        self.progress_rollout_train.close()

    def _train_actor_critic(self, level: int = -1, position: int = None) -> None:
        """Trains the actor and critic of the meta policy (level = -1) or, if specified, of
        the option."""
        if level == -1:  # meta-policy
            option = None
            policy = self.policy.meta_policy
            self._update_learning_rate(policy.optimizer, self.meta_learning_rate)
            evaluate_fn = policy.evaluate_actions
            pi_ent_coef = self.meta_pi_ent_coef
            vf_coef = self.meta_vf_coef
            clip_range_pi, clip_range_vf = self._get_current_meta_clip_ranges()
        else:
            option = self.policy.options_hierarchy[level][position]
            if not option.policy_trainable and not option.value_fn_trainable:
                return
            policy = option.get_policy()
            self._update_learning_rate(policy.optimizer, self.options_learning_rate)
            evaluate_fn = option.evaluate_actions
            pi_ent_coef = self.options_pi_ent_coef
            vf_coef = self.options_vf_coef
            clip_range_pi, clip_range_vf = self._get_current_options_clip_ranges()

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        approx_kl_divs = loss = None

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get_actor_critic_train_data(level, position, self.batch_size):
                options_actions = rollout_data.options_actions
                if isinstance(policy.action_space, spaces.Discrete):
                    options_actions = options_actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    policy.reset_noise(self.batch_size)

                values, log_prob, entropy = evaluate_fn(rollout_data.observations, options_actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if minibatch size == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = normalize_advantage(advantages)

                # ratio between old and new policy, should be one at the first iteration
                old_log_prob = rollout_data.old_log_probs
                ratio = th.exp(log_prob - old_log_prob)

                policy_loss, clip_fraction = ppo_loss(ratio, advantages, clip_range_pi)

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    old_values = rollout_data.old_values
                    values_pred = old_values + th.clamp(
                        values - old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                if option is not None:  # option
                    if not option.policy_trainable:
                        policy_loss = 0
                    if not option.value_fn_trainable:
                        value_loss = 0

                loss = policy_loss + pi_ent_coef * entropy_loss + vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_probs
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                policy.optimizer.step()

            self.progress_rollout_train.update(1)

            if not continue_training:
                break

        # Logs
        if loss is not None:
            # print(f" Loss: {loss:.3f}", end="")
            policy_name = "meta_policy" if level == -1 else get_option_name(level, position)
            base_str = f"{policy_name}/actor_critic/"
            self.logger.record(base_str + "loss", loss.item())
            self.logger.record(base_str + "policy_loss", np.mean(pg_losses))
            self.logger.record(base_str + "value_loss", np.mean(value_losses))
            self.logger.record(base_str + "entropy_loss", np.mean(entropy_losses))
            self.logger.record(base_str + "approx_kl", np.mean(approx_kl_divs))
            self.logger.record(base_str + "clip_fraction", np.mean(clip_fractions))
            values = self.rollout_buffer.get_values(level, position)
            returns = self.rollout_buffer.get_returns(level, position)
            explained_var = explained_variance(values.flatten(), returns.flatten())
            # assert not np.isnan(explained_var)
            self.logger.record(base_str + "explained_variance", explained_var)
            if hasattr(policy, "log_std"):
                self.logger.record(base_str + "std", th.exp(policy.log_std).mean().item())

    def _train_terminator(self, level: int, position: int) -> None:
        """Trains the terminator of each specified option."""
        option = self.policy.options_hierarchy[level][position]
        if not option.terminator_trainable:
            return
        terminator: Terminator = option.get_terminator()
        evaluate_fn = option.evaluate_terminations

        clip_range = self._get_current_terminator_clip_range()

        entropy_losses = []
        losses = []
        clip_fractions = []
        approx_kl_divs = loss = None

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get_terminator_train_data(level, position, self.batch_size):
                next_obs = rollout_data.next_observations

                # Evaluate terminations (indeed terminations, no continuations)
                terminations = th.ones(rollout_data.option_terminations.shape, device=self.device)
                tn_log_probs, entropy = evaluate_fn(next_obs, terminations)

                # Ratio between old and new termination log prob
                tn_ratio = th.exp(tn_log_probs - rollout_data.old_tn_log_probs)

                # Advantage estimate
                next_advantages = rollout_data.next_values - rollout_data.next_higher_level_value
                if self.normalize_advantage and len(next_advantages) > 1:
                    next_advantages = normalize_advantage(next_advantages)

                adjusted_advantage = th.Tensor(next_advantages) + self.options_tn_reg

                termination_loss, tn_clip_fraction = ppo_loss(tn_ratio, -adjusted_advantage, clip_range)

                entropy_loss = -th.mean(entropy)

                losses.append(termination_loss.item())
                clip_fractions.append(tn_clip_fraction)

                entropy_losses.append(entropy_loss.item())

                loss = termination_loss + self.options_tn_ent_coef * entropy_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = tn_log_probs - rollout_data.old_tn_log_probs
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                terminator.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(terminator.parameters(), self.max_grad_norm)
                terminator.optimizer.step()

            if not continue_training:
                break

        # Logs
        if loss is not None:
            policy_name = get_option_name(level, position)
            base_str = f"{policy_name}/terminator/"
            self.logger.record(base_str + "entropy_loss", np.mean(entropy_losses))
            self.logger.record(base_str + "termination_loss", np.mean(losses))
            self.logger.record(base_str + "approx_kl", np.mean(approx_kl_divs))
            self.logger.record(base_str + "clip_fraction", np.mean(clip_fractions))
            self.logger.record(base_str + "loss", loss.item())

    def _get_current_meta_clip_ranges(self) -> (float, float):
        # Compute current clip range
        clip_range = self.meta_pi_clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.meta_vf_clip_range is not None:
            clip_range_vf = self.meta_vf_clip_range(self._current_progress_remaining)
        else:
            clip_range_vf = None
        return clip_range, clip_range_vf

    def _get_current_options_clip_ranges(self) -> (float, float):
        # Compute current clip range
        clip_range = self.options_pi_clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.options_vf_clip_range is not None:
            clip_range_vf = self.options_vf_clip_range(self._current_progress_remaining)
        else:
            clip_range_vf = None
        return clip_range, clip_range_vf

    def _get_current_terminator_clip_range(self) -> float:
        return self.options_tn_clip_range(self._current_progress_remaining)

    def forward_all(self, obs: Union[np.ndarray, th.Tensor], *args, **kwargs):
        with th.no_grad():
            if isinstance(obs, np.ndarray):
                obs = obs_as_tensor(obs, self.device)
            (options, actions), values, log_probs = self.policy.forward_all(obs, *args, **kwargs)
        actions = actions.cpu().numpy().astype(self.action_space.dtype)
        return (options, actions), values, log_probs

    def forward_all_terminators(self, obs, *args, **kwargs):
        with th.no_grad():
            obs = obs_as_tensor(obs, self.device)
            return self.policy.forward_all_terminators(obs, *args, **kwargs)

    def _update_learning_rate(self, optimizers: Union[list[th.optim.Optimizer], th.optim.Optimizer],
                              lr_schedule: Schedule) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        # Log the current learning rate
        self.logger.record("hyperparameter_schedule/learning_rate",
                           lr_schedule(self._current_progress_remaining))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, lr_schedule(self._current_progress_remaining))

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy.meta_policy", "policy.options_hierarchy"]
        return state_dicts, []


def ppo_loss(ratio: th.Tensor, advantage: th.Tensor, clip_range: float) -> (th.Tensor, th.Tensor):
    """
    The clipped surrogate loss as proposed in the original PPO paper.
    :param ratio: for example, ratio between old and new policy pi/pi_old
    :param advantage: A
    :param clip_range: epsilon
    :return
        min(ratio * advantage, clip(ratio, 1 - eps, 1 + eps) * advantage) (scalar mean)
        the fraction of clipped entries
    """
    loss_part_1 = advantage * ratio
    loss_part_2 = advantage * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
    loss = -th.min(loss_part_1, loss_part_2).mean()

    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()

    return loss, clip_fraction


def normalize_advantage(advantages: th.tensor):
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


MODELS_BASE_PATH = "out/"


def load_agent(name: str = None,
               env_name: str = None,
               model_dir: str | Path = None,
               best_model: bool = True,
               n_envs: int = 1,
               reward_mode: str = None,
               render_mode: str = None,
               render_oc_overlay: bool = False,
               train: bool = False,
               verbose: int = 1,
               device: str = None):
    assert name is not None and env_name is not None or model_dir is not None

    if model_dir is None:
        env_identifier = get_env_identifier(env_name)
        model_dir = Path(MODELS_BASE_PATH, env_identifier, name)

    checkpoint_dir = model_dir / "checkpoints"
    if best_model:
        checkpoint_path = checkpoint_dir / "best_model.zip"
        vec_norm_path = model_dir / "checkpoints/best_vecnormalize.pkl"
    else:
        done_steps = get_most_recent_checkpoint_steps(checkpoint_dir)
        if done_steps is None:
            raise RuntimeError(f"No checkpoints found in '{checkpoint_dir.as_posix()}'.")
        checkpoint_path = checkpoint_dir / f"model_{done_steps}_steps.zip"
        vec_norm_path = checkpoint_dir / f"model_vecnormalize_{done_steps}_steps.pkl"
        print(f"Found most recent checkpoint '{checkpoint_path}'.")

    config_path = model_dir / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    if reward_mode is not None:
        config["environment"]["reward_mode"] = reward_mode

    hierarchy_shape = config["general"]["hierarchy_shape"]
    uses_options = len(hierarchy_shape) > 0

    env = init_vec_env(n_envs=n_envs,
                       seed=123,
                       vec_norm_path=vec_norm_path,
                       **config["environment"],
                       render_mode=render_mode,
                       render_oc_overlay=render_oc_overlay,
                       logic=config["meta_policy"]["logic"],
                       accept_predicates=not uses_options,
                       train=train)

    if device is None:
        device = config["device"]

    model = OptionsPPO.load(checkpoint_path,
                            env=env,
                            verbose=verbose,
                            render_mode=render_mode,
                            device=device,
                            custom_objects={"progress_total": None,
                                            "progress_rollout_train": None,
                                            "_last_option_terminations": None,
                                            "_last_active_options": None})

    return model
