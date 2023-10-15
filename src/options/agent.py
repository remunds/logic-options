from typing import Tuple, Any

import torch as th
from stable_baselines3.common.distributions import BernoulliDistribution
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy

from options.option import OptionCollection
from options.hierarchy import OptionsHierarchy


class OptionsAgent(BasePolicy):
    """The agent with options, containing the entire option hierarchy with all its levels
    and the meta policy.
    :param observation_space:
    :param action_space:
    :param lr_schedule:
    :param hierarchy_shape: List of options per hierarchy level; from highest to lowest level.
        Example: [2, 4, 8]. If empty, no options used => standard actor-critic PPO.
    :param symbolic_meta_policy: Whether the meta policy should be modeled with NUDGE or
        with an NN
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 lr_schedule,
                 hierarchy_shape: List[int],
                 symbolic_meta_policy: bool = False,
                 net_arch: List[int] = None,
                 **kwargs):
        super().__init__(observation_space=observation_space, action_space=action_space)

        if net_arch is None:
            net_arch = [64, 64]

        options_hierarchy = OptionsHierarchy(hierarchy_shape,
                                             observation_space,
                                             action_space,
                                             lr_schedule,
                                             net_arch)

        self.meta_policy = ActorCriticPolicy(observation_space,
                                             action_space=options_hierarchy.action_option_spaces[0],
                                             lr_schedule=lr_schedule,
                                             net_arch=net_arch,
                                             **kwargs)

        self.options_hierarchy = options_hierarchy
        self.symbolic_meta_policy = symbolic_meta_policy

    @property
    def hierarchy_size(self):
        return self.options_hierarchy.size

    @property
    def hierarchy_shape(self):
        return self.options_hierarchy.shape

    def get_option_by_id(self, option_id: th.Tensor) -> OptionCollection:
        """
        Turns a batch of option IDs into the corresponding list of options

        :param option_id: 2D matrix
        :return:
        """
        options = []
        for level, idx in option_id:
            options.append(self.options_hierarchy[level][idx])
        return OptionCollection(options)

    def set_training_mode(self, mode: bool) -> None:
        self.meta_policy.set_training_mode(mode)
        for level in self.options_hierarchy:
            for option in level:
                option.set_training_mode(mode)

    def reset_noise(self, n_envs: int = 1) -> None:
        self.meta_policy.reset_noise(n_envs)
        for level in self.options_hierarchy:
            for option in level:
                option.reset_noise(n_envs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        raise NotImplementedError()

    def forward_option(
            self,
            obs: th.Tensor,
            option_id: th.Tensor,
            deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in actor and critic for specified option.

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
            actions, values, log_probs = self.meta_policy(obs, deterministic)
            return (options, actions), values, log_probs.unsqueeze(1)

        actions = th.zeros(n_envs, *self.meta_policy.action_space.shape)
        values = th.zeros(n_envs, options.shape[1] + 1)
        log_probs = th.zeros(n_envs, options.shape[1] + 1)

        # Forward meta policy
        new_options, global_values, log_probs[:, 0] = self.meta_policy(obs, deterministic)
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
                active_option = self.options_hierarchy[level_id - 1][higher_level_option_id]

                option, values[env_id][level_id], log_probs[env_id][level_id] = \
                    active_option(env_obs, deterministic)

                # Replace terminated options
                if option_terminations[env_id][level_id]:
                    options[env_id][level_id] = option

            # Determine new action (always executed)
            lowest_level_option_id = options[env_id][-1]
            lowest_level_option = self.options_hierarchy[-1][lowest_level_option_id]
            actions[env_id], values[env_id, -1], log_probs[env_id, -1] = lowest_level_option(env_obs, deterministic)

        return (options, actions), values, log_probs

    def forward_option_terminator(
            self,
            obs: th.Tensor,
            option_id: th.Tensor,
            deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Forward in the terminator of a specified option."""
        option = self.get_option_by_id(option_id)
        return option.forward_terminator(obs, deterministic)

    def predict_all_values(self, obs: th.Tensor, option_traces: th.Tensor) -> th.Tensor:
        """Computes state-value for the global policy and each option as
        specified in the option trace."""
        values = th.zeros(option_traces.shape[0], option_traces.shape[1] + 1)
        values[:, 0] = self.meta_policy.predict_values(obs).squeeze()
        for env_id, trace in enumerate(option_traces):
            env_obs = th.unsqueeze(obs[env_id], dim=0)
            for level_id, option_id in enumerate(trace):
                option = self.options_hierarchy[level_id][option_id]
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
                option = self.options_hierarchy[level][option_id]
                termination, log_prob = option.forward_terminator(env_obs, deterministic)
                terminations[env_id, level] = termination
                log_probs[env_id, level] = log_prob

        return terminations, log_probs

    def get_option_termination_dist(self, obs: th.Tensor, option_id: th.Tensor) -> BernoulliDistribution:
        option = self.get_option_by_id(option_id)
        return option.get_termination_dist(obs)

    def evaluate_option_terminations(
            self,
            obs: th.Tensor,
            option_id: th.Tensor,
            terminations: th.Tensor
    ) -> th.Tensor:
        option = self.get_option_by_id(option_id)
        return option.evaluate_terminations(obs=obs, terminations=terminations)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = dict(options_hierarchy=self.hierarchy_shape,
                    symbolic_meta_policy=self.symbolic_meta_policy)
        return data

    def to(self, device):
        self.meta_policy = self.meta_policy.to(device)
        for l, level in enumerate(self.options_hierarchy):
            for o, option in enumerate(level):
                self.options_hierarchy[l][o] = option.to(device)
        return self
