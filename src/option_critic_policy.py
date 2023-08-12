from typing import Tuple

import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import BernoulliDistribution
from gymnasium.spaces import Discrete, Space

from option import Option, OptionCollection


class GlobalOptionsPolicy(ActorCriticPolicy):
    """Policy representing an entirety of an option hierarchy with all its levels.
    Acts itself as the global (inter-option) policy."""

    def __init__(self,
                 observation_space,
                 action_space,
                 lr_schedule,
                 options_hierarchy: list[int],
                 device: th.device,
                 **kwargs):
        """
        :param observation_space:
        :param action_space:
        :param lr_schedule:
        :param options_hierarchy: List of options per hierarchy level; from highest to lowest level.
            Example: [2, 4, 8]. If empty, no options used => standard actor-critic PPO.
        :param device:
        """

        action_option_spaces = []
        for n_options in options_hierarchy:
            action_option_spaces.append(Discrete(n_options))
        action_option_spaces.append(action_space)

        super().__init__(observation_space,
                         action_space=action_option_spaces[0],
                         lr_schedule=lr_schedule, **kwargs)

        self.hierarchy_size = len(options_hierarchy)  # 0 => no options, 1 => one level of options...
        self.options_hierarchy = options_hierarchy

        self.action_option_spaces = action_option_spaces

        def make_option(local_action_space: Space):
            return Option(observation_space=observation_space,
                          action_space=local_action_space,
                          lr_schedule=lr_schedule,
                          device=device,
                          **kwargs)

        self.options = []  # higher-level options first, lower-level options last
        for h, n_options in enumerate(self.options_hierarchy):
            assert n_options > 1, "It doesn't make sense to have a layer containing only one option."
            action_option_space = self.action_option_spaces[h+1]
            level_options = [make_option(action_option_space) for _ in range(n_options)]
            self.options.append(level_options)

    def get_option_by_id(self, option_id: th.Tensor) -> OptionCollection:
        """
        Turns a batch of option IDs into the corresponding list of options

        :param option_id: 2D matrix
        :return:
        """
        options = []
        for level, idx in option_id:
            options.append(self.options[level][idx])
        return OptionCollection(options)

    def set_training_mode(self, mode: bool) -> None:
        super().set_training_mode(mode)
        for level in self.options:
            for option in level:
                option.set_training_mode(mode)

    def reset_noise(self, n_envs: int = 1) -> None:
        super().reset_noise(n_envs)
        for level in self.options:
            for option in level:
                option.reset_noise(n_envs)

    def forward_option(
            self,
            obs: th.Tensor,
            option_id: th.Tensor,
            deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in actor and critic on global level.

        :param obs: Observation
        :param option_id: The target option's index [level, ID]
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value, and log probability of the action
        """
        option = self.get_option_by_id(option_id)
        return option.forward(obs, deterministic)

    def forward_all(
            self,
            obs: th.Tensor,
            option_traces: th.Tensor,
            option_terminations: th.Tensor,
            deterministic: bool = False
    ) -> ((th.Tensor, th.Tensor), th.Tensor, th.Tensor):
        """
        Forward pass in all actors and critics. Terminated options will be replaced by a new
        option.

        :param obs: Observation
        :param option_traces:
        :param option_terminations: Same shape as option_traces where each entry is True iff the
            corresponding option in option_traces has terminated.
        :param deterministic:
        :return: option trace with action appended, corresponding values, and log probabilities
        """
        n_envs = len(obs)
        options = option_traces.clone()

        if self.hierarchy_size == 0:
            actions, values, log_probs = self(obs, deterministic)
            return (options, actions), values, log_probs.unsqueeze(1)

        actions = th.zeros(n_envs, *self.action_space.shape)
        values = th.zeros(n_envs, options.shape[1] + 1)
        log_probs = th.zeros(n_envs, options.shape[1] + 1)

        # Forward global policy
        new_options, global_values, log_probs[:, 0] = self(obs, deterministic)
        values[:, 0] = global_values.squeeze()
        is_terminated = option_terminations[:, 0]
        options[is_terminated, 0] = new_options[is_terminated]

        for env_id, trace in enumerate(option_traces):
            env_obs = th.unsqueeze(obs[env_id], dim=0)

            # Determine new options (if needed)
            for level_id, option_id in enumerate(trace):
                if level_id == 0:
                    continue

                higher_level_option_id = options[env_id][level_id - 1]
                active_option = self.options[level_id - 1][higher_level_option_id]

                option, values[env_id][level_id], log_probs[env_id][level_id] = \
                    active_option(env_obs, deterministic)

                # Replace terminated options
                if option_terminations[env_id][level_id]:
                    options[env_id][level_id] = option

            # Determine new action (always executed)
            lowest_level_option_id = options[env_id][-1]
            lowest_level_option = self.options[-1][lowest_level_option_id]
            actions[env_id], values[env_id, -1], log_probs[env_id, -1] = lowest_level_option(env_obs, deterministic)

        return (options, actions), values, log_probs

    def forward_option_terminator(
            self,
            obs: th.Tensor,
            option_id: th.Tensor,
            deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor]:
        option = self.get_option_by_id(option_id)
        return option.forward_terminator(obs, deterministic)

    def predict_all_values(self, obs: th.Tensor, option_traces: th.Tensor) -> th.Tensor:
        """Computes state-value for the global policy and each option as
        specified in the option trace."""
        values = th.zeros(option_traces.shape[0], option_traces.shape[1] + 1)
        values[:, 0] = self.predict_values(obs).squeeze()
        for env_id, trace in enumerate(option_traces):
            env_obs = th.unsqueeze(obs[env_id], dim=0)
            for level_id, option_id in enumerate(trace):
                option = self.options[level_id][option_id]
                values[env_id, level_id + 1] = option.predict_values(env_obs)
        return values

    def forward_all_terminators(
            self,
            obs: th.Tensor,
            option_traces: th.Tensor,
            deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor]:

        n_envs = len(obs)

        terminations = th.zeros((n_envs, self.hierarchy_size), dtype=th.bool)
        log_probs = th.zeros((n_envs, self.hierarchy_size), dtype=th.float)

        for env_id, trace in enumerate(option_traces):
            env_obs = th.unsqueeze(obs[env_id], dim=0)
            for level, option_id in enumerate(trace):
                option = self.options[level][option_id]
                termination, log_prob = option.forward_terminator(env_obs, deterministic)
                terminations[env_id, level] = termination
                log_probs[env_id, level] = log_prob

        return terminations, log_probs

    # def choose_option(self, obs: th.Tensor, option_id: th.Tensor, deterministic: bool = False) -> th.Tensor:
    #     option_id = th.clone(option_id)
    #     while len(option_id) < self.hierarchy_height:
    #         option = self.get_option_by_id(option_id)
    #         action, _, _ = option.forward(obs, deterministic)
    #         option_id = th.cat([option_id, action.squeeze()])
    #     return option_id

    def get_option_termination_dist(self, obs: th.Tensor, option_id: th.Tensor) -> BernoulliDistribution:
        option = self.get_option_by_id(option_id)
        return option.get_termination_dist(obs)
    #
    # def predict_option_termination(self, obs: th.Tensor, option_id: th.Tensor) -> th.Tensor:
    #     option = self.get_option_by_id(option_id)
    #     return option.predict_termination(obs)

    def evaluate_option_terminations(
            self,
            obs: th.Tensor,
            option_id: th.Tensor,
            terminations: th.Tensor
    ) -> th.Tensor:
        option = self.get_option_by_id(option_id)
        return option.evaluate_terminations(obs=obs, terminations=terminations)
