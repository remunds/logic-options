from typing import Optional, Tuple, Union

import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from gymnasium.spaces import Discrete

from option import Option, OptionCollection


class OptionCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy where actions are replaced by options. All options are
    an AC policy on their own. The policy of this class is the inter-option policy."""

    def __init__(self,
                 observation_space,
                 action_space,
                 lr_schedule,
                 n_options: int,
                 n_envs: int,
                 device: th.device,
                 **kwargs):
        self.option_space = Discrete(n_options)

        super().__init__(observation_space,
                         action_space=self.option_space,
                         lr_schedule=lr_schedule, **kwargs)

        self.n_options = n_options
        self.options = OptionCollection([Option(observation_space=observation_space,
                                                action_space=action_space,
                                                lr_schedule=lr_schedule,
                                                device=device,
                                                **kwargs) for _ in range(n_options)])
        self.active_option_id = th.LongTensor(n_envs * [0])

    def set_active_option_id(self, option_id: th.Tensor, where: th.Tensor = None) -> None:
        if where is not None:
            assert len(self.active_option_id) == len(where)
            assert len(option_id) == th.sum(where)
            self.active_option_id[where] = option_id
        else:
            self.active_option_id = option_id

    def get_active_option_id(self) -> th.Tensor:
        return self.active_option_id

    def get_active_option(self) -> Union[Option, OptionCollection]:
        return self.options[self.get_active_option_id()]

    def set_training_mode(self, mode: bool) -> None:
        super().set_training_mode(mode)
        self.options.set_training_mode(mode)

    def reset_noise(self, n_envs: int = 1) -> None:
        super().reset_noise(n_envs)
        self.options.reset_noise(n_envs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in actor and critic for currently active option.

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value, and log probability of the action
        """
        active_option = self.get_active_option()
        return active_option.forward(obs=obs, deterministic=deterministic)

    def forward_terminator(self, obs: th.Tensor, deterministic: bool = False) -> (th.Tensor, th.Tensor):
        """
        Forward pass through termination network for currently active option.

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: option terminations and their log probabilities
        """
        active_option = self.get_active_option()
        return active_option.forward_terminator(obs, deterministic)

    def get_option_distribution(self, obs: th.Tensor) -> Distribution:
        return super().get_distribution(obs)

    def evaluate_options(self, obs: th.Tensor, options: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        return super().evaluate_actions(obs=obs, actions=options)

    def evaluate_terminations(
            self,
            obs: th.Tensor,
            terminations: th.Tensor
    ) -> th.Tensor:
        """
        Evaluate terminations according to the current policy, given the observations.

        :param obs: Observations
        :param terminations: Terminations
        :return: estimated termination log likelihood
        """
        active_option = self.get_active_option()
        return active_option.evaluate_terminations(obs=obs, terminations=terminations)

    def choose_option(self, obs: th.Tensor) -> th.Tensor:
        return self.get_option_distribution(obs).get_actions()

    # def choose_termination(self, obs: th.Tensor) -> th.Tensor:
    #     return self.get_termi
