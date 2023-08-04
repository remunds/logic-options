from typing import Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, explained_variance
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv
from stable_baselines3.ppo.ppo import PPO
from torch.nn import functional as F

from option_critic_policy import GlobalOptionsPolicy
from options_rollout_buffer import OptionsRolloutBuffer


class OptionsPPO(PPO):
    """
    Proximal Policy Optimization algorithm (PPO) with options

    :param options_hierarchy: Number of options
    :param termination_regularizer: Variable xi. If increased, option termination
            probability gets decreased, i.e., longer options are encouraged. Set
            to 0 do disable this regularization.
    :param **kwargs: PPO keyworded arguments, see SB3 docs:
        https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    """

    rollout_buffer: OptionsRolloutBuffer
    policy: GlobalOptionsPolicy

    def __init__(self,
                 env: SubprocVecEnv,
                 options_policy: Type[GlobalOptionsPolicy],
                 options_hierarchy: list[int],
                 device: th.device,
                 termination_regularizer: float = 0,
                 **kwargs):
        policy_kwargs = {"options_hierarchy": options_hierarchy,
                         "device": device,
                         "net_arch": [64, 64]}  # TODO: Test this hyperparameter
        super().__init__(policy=options_policy,
                         policy_kwargs=policy_kwargs,
                         env=env,
                         device=device,
                         **kwargs)

        self.termination_regularizer = termination_regularizer
        self.hierarchy_size = self.policy.hierarchy_size

        self._last_option_terminations = th.ones(env.num_envs, self.policy.hierarchy_size).type(th.BoolTensor)
        self._last_active_options = th.zeros(env.num_envs, self.policy.hierarchy_size).type(th.LongTensor)

        self.rollout_buffer = OptionsRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            option_hierarchy_size=len(options_hierarchy),
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

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
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            with th.no_grad():
                new_obs_tensor = obs_as_tensor(self._last_obs, self.device)
                terminations, tn_log_probs = self.policy.forward_all_terminators(new_obs_tensor, options)

                # If a higher-level option exits, all lower-level options exit, too
                for level in range(1, self.policy.hierarchy_size):
                    terminations[:, level] |= terminations[:, level - 1]

                # TODO: efficiency can be improved as next values are needed only for terminated options
                next_values = self.policy.predict_all_values(new_obs_tensor, self._last_active_options)

            # Save observed data to rollout buffer
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                next_values,
                log_probs,
                options,
                terminations,
                tn_log_probs,
            )

            # Prepare next iteration
            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_active_options = options
            terminations[dones] = True
            self._last_option_terminations = terminations

        # Compute value for the last timestep
        with th.no_grad():
            values = self.policy.predict_all_values(new_obs_tensor, options)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Train global policy
        self._train_actor_critic()

        # Train actor, critic, and terminator of each option
        for level, n_options in enumerate(self.policy.options_hierarchy):
            for option_id in range(n_options):
                self._train_actor_critic(level, option_id)
                self._train_terminator(level, option_id)

    def _train_actor_critic(self, level: int = None, option_id: int = None) -> None:
        """Trains the actor and critic of the global policy or, if specified, of
        the option."""
        if level is None or option_id is None:
            policy = self.policy
        else:
            policy = self.policy.options[level][option_id]

        clip_range, clip_range_vf = self._get_current_clip_ranges()

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        approx_kl_divs = loss = None

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get_actor_critic_train_data(level, option_id, self.batch_size):
                options_actions = rollout_data.options_actions.long().flatten().unsqueeze(1)

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    policy.reset_noise(self.batch_size)

                values, log_prob, entropy = policy.evaluate_actions(rollout_data.observations, options_actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                old_log_prob = rollout_data.old_log_probs
                ratio = th.exp(log_prob - old_log_prob)

                policy_loss, clip_fraction = ppo_loss(ratio, advantages, clip_range)

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
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

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

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        policy_name = "global_policy" if level is None or option_id is None else f"option_{level}.{option_id}"
        base_str = f"train/{policy_name}/actor-critic/"
        self.logger.record(base_str + "entropy_loss", np.mean(entropy_losses))
        self.logger.record(base_str + "policy_loss", np.mean(pg_losses))
        self.logger.record(base_str + "value_loss", np.mean(value_losses))
        self.logger.record(base_str + "approx_kl", np.mean(approx_kl_divs))
        self.logger.record(base_str + "clip_fraction", np.mean(clip_fractions))
        if loss is not None:
            self.logger.record(base_str + "loss", loss.item())
        self.logger.record(base_str + "explained_variance", explained_var)
        if hasattr(policy, "log_std"):
            self.logger.record(base_str + "std", th.exp(policy.log_std).mean().item())

        self.logger.record(base_str + "n_updates", self._n_updates, exclude="tensorboard")  # FIXME
        self.logger.record(base_str + "clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record(base_str + "clip_range_vf", clip_range_vf)

    def _train_terminator(self, level: int, option_id: int) -> None:
        """Trains the terminator of each specified option."""
        policy = self.policy.options[level][option_id]

        clip_range, clip_range_vf = self._get_current_clip_ranges()

        entropy_losses = []
        losses = []
        clip_fractions = []
        approx_kl_divs = loss = None

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get_terminator_train_data(level, option_id, self.batch_size):
                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    policy.reset_noise(self.batch_size)

                obs = rollout_data.observations

                terminations = rollout_data.option_terminations
                tn_log_probs, entropy = policy.evaluate_terminations(obs, terminations)

                # Ratio between old and new termination log prob
                tn_ratio = th.exp(tn_log_probs - rollout_data.old_tn_log_probs)

                # Advantage estimate
                next_advantages = rollout_data.next_higher_level_value - rollout_data.next_values
                adjusted_advantage = th.Tensor(next_advantages) + self.termination_regularizer

                termination_loss, tn_clip_fraction = ppo_loss(tn_ratio, -adjusted_advantage, clip_range)

                entropy_loss = -th.mean(entropy)

                losses.append(termination_loss.item())
                clip_fractions.append(tn_clip_fraction)

                entropy_losses.append(entropy_loss.item())

                loss = termination_loss + self.ent_coef * entropy_loss

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
                policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # Logs
        policy_name = f"option_{level}.{option_id}"
        base_str = f"train/{policy_name}/terminator/"
        self.logger.record(base_str + "entropy_loss", np.mean(entropy_losses))
        self.logger.record(base_str + "termination_loss", np.mean(losses))
        self.logger.record(base_str + "approx_kl", np.mean(approx_kl_divs))
        self.logger.record(base_str + "clip_fraction", np.mean(clip_fractions))
        if loss is not None:
            self.logger.record(base_str + "loss", loss.item())
        if hasattr(policy, "log_std"):
            self.logger.record(base_str + "std", th.exp(policy.log_std).mean().item())

        self.logger.record(base_str + "n_updates", self._n_updates, exclude="tensorboard")  # FIXME
        self.logger.record(base_str + "clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record(base_str + "clip_range_vf", clip_range_vf)

    def _get_current_clip_ranges(self) -> (float, float):
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        else:
            clip_range_vf = None
        return clip_range, clip_range_vf

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
    loss = th.min(loss_part_1, loss_part_2).mean()

    clip_fraction = -th.mean((th.abs(ratio - 1) > clip_range).float()).item()

    return loss, clip_fraction
