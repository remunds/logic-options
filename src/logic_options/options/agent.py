import os.path
from typing import Tuple, Any, List, Dict

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import CategoricalDistribution, Distribution
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from logic_options.options.option import OptionCollection, Option
from logic_options.options.hierarchy import OptionsHierarchy
from logic_options.logic.policy import NudgePolicy
from logic_options.options.meta_policy import MetaPolicy


class OptionsAgent(BasePolicy):
    """The agent with options, containing the entire option hierarchy with all its levels
    and the meta policy.
    :param observation_space:
    :param action_space:
    :param lr_schedule:
    :param hierarchy_shape: List of options per hierarchy level; from highest to lowest level.
        Example: [2, 4, 8]. If empty, no options used => standard actor-critic PPO.
    :param logic_meta_policy: Whether the meta policy should be modeled with NUDGE or
        with an NN
    :param components: List of information which option to initialize with which
        already-trained component for transfer learning.
    :param accepts_predicates: If True, treats top-level option IDs as predicates, hence,
        converts them to actual option IDs before usage.
    """

    # meta_policy: ActorCriticPolicy
    meta_policy: MetaPolicy
    options_hierarchy: OptionsHierarchy

    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 lr_schedule,
                 hierarchy_shape: List[int],
                 logic_meta_policy: bool = False,
                 net_arch: List[int] = None,
                 env_name: str = None,
                 accepts_predicates: bool = False,
                 device: str = None,
                 **kwargs):
        super().__init__(observation_space=observation_space, action_space=action_space)

        if net_arch is None:
            net_arch = [64, 64]

        self.lr_schedule = lr_schedule
        self.hierarchy_shape = hierarchy_shape
        self.logic_meta_policy = logic_meta_policy
        self.net_arch = net_arch
        self.env_name = env_name
        self.accepts_predicates = accepts_predicates
        self.pretrained_options = None

        options_hierarchy = OptionsHierarchy(hierarchy_shape,
                                             observation_space,
                                             action_space,
                                             lr_schedule,
                                             net_arch)

        if self.logic_meta_policy:
            if len(self.hierarchy_shape) > 0:
                action_space = None
            self.meta_policy = NudgePolicy(
                env_name=env_name,
                observation_space=observation_space,
                action_space=action_space,
                lr_schedule=lr_schedule,
                net_arch=net_arch,
                uses_options=self.accepts_predicates,
                device=device,
                **kwargs
            )
        else:
            # self.meta_policy = ActorCriticPolicy(
            self.meta_policy = MetaPolicy(
                observation_space=observation_space,
                action_space=options_hierarchy.action_option_spaces[0],
                lr_schedule=lr_schedule,
                net_arch=net_arch,
                **kwargs
            )

        self.options_hierarchy = options_hierarchy

        if self.accepts_predicates:
            self._init_predicate_conversion(device)

    def _init_predicate_conversion(self, device):
        """Computes the pred2option np.array which converts predicate IDs into top-level
        option position IDs."""

        assert isinstance(self.meta_policy, NudgePolicy)
        n_top_level_options = self.hierarchy_shape[0]
        pred2option = []

        # Find the corresponding top-level-option for each predicate
        for predicate in self.meta_policy.predicates:
            for o in range(n_top_level_options):
                if f"opt{o}" in predicate:
                    pred2option.append(o)
                    break
                elif o + 1 == n_top_level_options:
                    raise ValueError(f"Invalid predicate defined! The predicate '{predicate}' does "
                                     f"not contain the identifier of any option. It must contain a "
                                     f"substring of the form 'opt{o}' for, e.g., option {o}.")
        self.pred2option = th.tensor(pred2option, device=device)

    def load_pretrained_options(self,
                                configuration: List[Dict[str, Any]] = None,
                                env: VecEnv = None,
                                device="auto"):
        if configuration is None:
            return

        self.pretrained_options = configuration
        from logic_options.options.ppo import OptionsPPO

        for component in configuration:
            level = component["level"]
            position = component["position"]
            model_path = component["model_path"]
            policy_trainable = component["policy_trainable"]
            value_fn_trainable = component["value_fn_trainable"]
            terminator_trainable = component["terminator_trainable"]

            # Load vec normalize if exists
            vec_norm_path = model_path + ".pkl"
            if env is not None and os.path.exists(vec_norm_path):
                env = VecNormalize.load(vec_norm_path, venv=env)
                vec_norm = env
            else:
                vec_norm = None

            # Load actor-critic policy
            ppo = OptionsPPO.load(model_path + ".zip", env, custom_objects={"progress_rollout_train": None,
                                                                            "progress_total": None})
            policy: ActorCriticPolicy = ppo.policy.meta_policy

            old_option = self.options_hierarchy.options[level][position]

            # Initialize option, re-use old terminator as it might be a trained one
            # TODO: also enable to re-use the old value function
            option = Option(policy=policy,
                            policy_trainable=policy_trainable,
                            value_fn_trainable=value_fn_trainable,
                            terminator=old_option.get_terminator(),
                            terminator_trainable=terminator_trainable,
                            vec_norm=vec_norm,
                            lr_schedule=self.lr_schedule)
            option = option.to(device)
            self.options_hierarchy.options[level][position] = option

            pi_trainable = "trainable" if policy_trainable else "untrainable"
            tn_trainable = "trainable" if terminator_trainable else "untrainable"
            print(f"Loaded pre-trained model from '{model_path}' as option "
                  f"on level {level} at position {position} with "
                  f"{pi_trainable} actor-critic and {tn_trainable} terminator.")

    @property
    def hierarchy_size(self):
        return self.options_hierarchy.size

    @property
    def n_policies(self):
        return np.sum(self.hierarchy_shape) + 1

    def preds2options(self, predicates: th.Tensor) -> th.Tensor:
        """Converts predicates to top-level option positions. Needed when
        the meta policy is logical."""
        if self.accepts_predicates:
            options = self.pred2option[predicates]
        else:
            options = predicates
        return options

    def get_option_by_idx(self, index: th.Tensor) -> OptionCollection:
        """
        Turns a batch of option indexes into the corresponding list of options
        :param index: 2D matrix
        """
        options = []
        for level, position in index:
            if level == 0:
                position = self.preds2options(position)
            options.append(self.options_hierarchy[level][position])
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
            option_idx: th.Tensor,
            deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in actor and critic for specified option.

        :param obs: Observation
        :param option_idx: The target option's index [level, position]
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value, and log probability of the action
        """
        option = self.get_option_by_idx(option_idx)
        return option.forward(obs, deterministic)
    
    def _get_terminations(self, option_distribution: Distribution, deterministic: bool = False) -> th.Tensor:
        best_option = option_distribution.get_actions(deterministic)


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
        :param option_traces: For each hierarchy level the position of the currently active option.
        :param option_terminations: Same shape as option_traces where each entry is True iff the
            corresponding option in option_traces has terminated.
        :param deterministic:
        :return: option trace with action appended, corresponding values, and log probabilities
        """
        n_envs = len(obs)
        option_traces = option_traces.clone().type(th.long)

        if self.hierarchy_size == 0:
            actions, values, log_probs = self.meta_policy(obs, deterministic)
            return (option_traces, actions), values, log_probs.unsqueeze(1)

        actions = th.zeros(n_envs, *self.meta_policy.action_space.shape)
        values = th.zeros(n_envs, option_traces.shape[1] + 1)
        log_probs = th.zeros(n_envs, option_traces.shape[1] + 1)

        # Forward meta policy and replace top-level option if needed

        # TODO: modify to get log_probs of all options

        #TODO: For all policies (meta + options): use distribution 
        # to decide if next level option should terminate (based on highest log_prob)

        # new_options, global_values, log_probs[:, 0] = self.meta_policy(obs, deterministic)
        global_values, option_distribution = self.meta_policy(obs, deterministic)
        new_options = option_distribution.get_actions(deterministic)

        is_terminated = option_terminations[:, 0]
        option_traces[is_terminated, 0] = new_options[is_terminated]  # is predicate if meta_policy is logic
        values[:, 0] = global_values.squeeze()

        for env in range(n_envs):
            env_obs = th.unsqueeze(obs[env], dim=0)

            # Compute values for each level and, if needed, determine new options
            for level in range(0, self.hierarchy_size - 1):
                option_pos = option_traces[env][level]
                if level == 0:
                    option_pos = self.preds2options(option_pos)

                option = self.options_hierarchy[level][option_pos]

                new_lower_level_option_pos, values[env][level + 1], log_probs[env][level + 1] = \
                    option(env_obs, deterministic)

                # Replace terminated lower-level option
                if option_terminations[env][level + 1]:
                    option_traces[env][level + 1] = new_lower_level_option_pos

            # Determine new action (always executed)
            lowest_level_option_pos = self.preds2options(option_traces[env][-1])
            lowest_level_option = self.options_hierarchy[-1][lowest_level_option_pos]
            actions[env], values[env, -1], log_probs[env, -1] = lowest_level_option(env_obs, deterministic)

        return (option_traces, actions), values, log_probs

    def forward_option_terminator(
            self,
            obs: th.Tensor,
            option_idx: th.Tensor,
            deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Forward in the terminator of specified options."""
        options = self.get_option_by_idx(option_idx)
        return options.forward_terminator(obs, deterministic)

    def predict_all_values(self, obs: th.Tensor, option_traces: th.Tensor) -> th.Tensor:
        """Computes state-value for the global policy and each option as
        specified in the option trace."""

        if self.hierarchy_size == 0:
            return self.meta_policy.predict_values(obs).squeeze().unsqueeze(-1)

        option_traces = option_traces.clone().type(th.long)
        option_traces[:, 0] = self.preds2options(option_traces[:, 0])

        values = th.zeros(option_traces.shape[0], option_traces.shape[1] + 1)
        values[:, 0] = self.meta_policy.predict_values(obs).squeeze()
        for env, trace in enumerate(option_traces):
            env_obs = th.unsqueeze(obs[env], dim=0)
            for level, position in enumerate(trace):
                option = self.options_hierarchy[level][position]
                values[env, level + 1] = option.predict_values(env_obs)
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

        if self.hierarchy_size == 0:
            return terminations, log_probs

        option_traces = option_traces.clone().type(th.long)
        option_traces[:, 0] = self.preds2options(option_traces[:, 0])

        for env_id, trace in enumerate(option_traces):
            env_obs = th.unsqueeze(obs[env_id], dim=0)
            for level, position in enumerate(trace):
                option = self.options_hierarchy[level][position]
                termination, log_prob = option.forward_terminator(env_obs, deterministic)
                terminations[env_id, level] = termination
                log_probs[env_id, level] = log_prob

        return terminations, log_probs
    
    def forward_new_terminator(
            self,
            obs: th.Tensor,
            option_traces: th.Tensor,
            deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Use log_prob of actions of all options to determine whether 
        current option should terminate (because another option is better). 
        """
        # get all log_probs
        n_envs = len(obs)
        option_traces = option_traces.clone().type(th.long)
        option_traces[:, 0] = self.preds2options(option_traces[:, 0])
        log_probs = th.zeros((n_envs, self.hierarchy_size), dtype=th.float)
        for env_id, trace in enumerate(option_traces):
            env_obs = th.unsqueeze(obs[env_id], dim=0)
            for level, position in enumerate(trace):
                option = self.options_hierarchy[level][position]
                _, log_prob = option(env_obs, deterministic)
                log_probs[env_id, level] = log_prob
        #TODO
        # get highest log_prob for each option

        # check if highest log_prob is higher than current log_prob
        # if yes, terminate current option and start new option




    def get_option_termination_dist(self, obs: th.Tensor, option_idx: th.Tensor) -> CategoricalDistribution:
        option = self.get_option_by_idx(option_idx)
        return option.get_termination_dist(obs)

    def evaluate_terminations(
            self,
            obs: th.Tensor,
            option_traces: th.Tensor,
            terminations: th.Tensor
    ) -> th.Tensor:
        n_envs = len(obs)

        log_probs = th.zeros((n_envs, self.hierarchy_size), dtype=th.float)

        if self.hierarchy_size == 0:
            return log_probs

        option_traces = option_traces.clone().type(th.long)
        option_traces[:, 0] = self.preds2options(option_traces[:, 0])

        for env_id, trace in enumerate(option_traces):
            env_obs = th.unsqueeze(obs[env_id], dim=0)
            term = th.unsqueeze(terminations[env_id], dim=0)
            for level, position in enumerate(trace):
                option = self.options_hierarchy[level][position]
                log_prob, _ = option.evaluate_terminations(env_obs, term)
                log_probs[env_id, level] = log_prob

        return log_probs

    def evaluate_option_terminations(
            self,
            obs: th.Tensor,
            option_idx: th.Tensor,
            terminations: th.Tensor
    ) -> th.Tensor:
        option = self.get_option_by_idx(option_idx)
        return option.evaluate_terminations(obs=obs, terminations=terminations)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(dict(
            options_hierarchy=self.hierarchy_shape,
            symbolic_meta_policy=self.logic_meta_policy,
            lr_schedule=self.lr_schedule,
            hierarchy_shape=self.hierarchy_shape,
            net_arch=self.net_arch,
            env_name=self.env_name,
            accepts_predicates=self.accepts_predicates,
            device=self.device,
            components=self.pretrained_options,
        ))
        return data

    def to(self, device):
        self.meta_policy = self.meta_policy.to(device)
        for level in self.options_hierarchy:
            for option in level:
                option.to(device)
        return self
