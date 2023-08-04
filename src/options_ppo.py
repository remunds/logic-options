from typing import Type

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, explained_variance
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv
from stable_baselines3.ppo.ppo import PPO
from torch.nn import functional as F

from option_critic_policy import GlobalOptionsPolicy
from options_rollout_buffer import OptionsRolloutBuffer, OptionsRolloutBufferSamples


class OptionsPPO(PPO):
    """
    Proximal Policy Optimization algorithm (PPO) with options

    :param options_hierarchy: Number of options
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
                 **kwargs):
        policy_kwargs = {"options_hierarchy": options_hierarchy,
                         "device": device,
                         "net_arch": [64, 64]}  # TODO: Test this hyperparameter
        super().__init__(policy=options_policy,
                         policy_kwargs=policy_kwargs,
                         env=env,
                         device=device,
                         **kwargs)
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
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        else:
            clip_range_vf = None

        entropy_losses = []
        pg_losses, value_losses, tn_losses = [], [], []
        pi_clip_fractions = []
        tn_clip_fractions = []
        approx_kl_divs = loss = None

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # Activate options
                options = rollout_data.options
                self.policy.set_active_option_id(options)

                obs = rollout_data.observations
                terminations = rollout_data.terminations

                values, pi_log_prob, entropy = self.policy.evaluate_actions(obs, actions)
                values = values.flatten()

                tn_log_prob = self.policy.evaluate_terminations(obs, terminations)

                # Normalize advantage (makes sense only if minibatch size > 1)
                action_advantages = rollout_data.advantages  # TODO: useful in case of options? Affects multiple models at once
                if self.normalize_advantage and len(action_advantages) > 1:
                    action_advantages = (action_advantages - action_advantages.mean()) / (action_advantages.std() + 1e-8)

                option_advantages = ...  # TODO

                # Ratio between old and new, should both be one at the first iteration
                pi_ratio = th.exp(pi_log_prob - rollout_data.old_pi_log_prob)  # old and new policy
                tn_ratio = th.exp(tn_log_prob - rollout_data.old_tn_log_prob)  # old and new termination

                # clipped surrogate loss
                policy_loss, pi_clip_fraction = ppo_loss(pi_ratio, action_advantages, clip_range)
                termination_loss, tn_clip_fraction = ppo_loss(tn_ratio, -option_advantages, clip_range)
                # TODO: insert termination regularization xi

                # Logging
                pg_losses.append(policy_loss.item())
                tn_losses.append(termination_loss.item())
                pi_clip_fractions.append(pi_clip_fraction)
                tn_clip_fractions.append(tn_clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-pi_log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + termination_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = pi_log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/termination_loss", np.mean(tn_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/pi_clip_fraction", np.mean(pi_clip_fractions))
        self.logger.record("train/tn_clip_fraction", np.mean(tn_clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


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
