from __future__ import annotations

from typing import Type, Union, Optional, Dict, List, Tuple

import numpy as np
import torch as th
from gymnasium.spaces import Space
from stable_baselines3.common.distributions import BernoulliDistribution
from stable_baselines3.common.policies import ActorCriticPolicy, BaseModel
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import FlattenExtractor
from torch import nn

from utils.common import get_net_from_layer_dims


class Option:
    """AC model extended by a flow for termination prediction."""

    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 lr_schedule: Schedule,
                 net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                 **kwargs):
        self.policy = ActorCriticPolicy(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=FlattenExtractor,
            features_extractor_kwargs=None,
            share_features_extractor=True,
            **kwargs
        )

        tn_net_arch = net_arch["tn"] if isinstance(net_arch, dict) else self.policy.net_arch["pi"]
        self.terminator = Terminator(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor=self.policy.features_extractor,
            net_arch=tn_net_arch,
            activation_fn=self.policy.activation_fn,
            optimizer_class=self.policy.optimizer_class,
            optimizer_kwargs=self.policy.optimizer_kwargs,
            normalize_images=self.policy.normalize_images,
        )

        self.observation_space = observation_space
        self.action_space = action_space

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        return self.policy.forward(obs, deterministic)

    def set_training_mode(self, mode: bool):
        self.policy.set_training_mode(mode)
        self.terminator.set_training_mode(mode)

    def reset_noise(self, n_envs: int = 1):
        self.policy.reset_noise(n_envs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Terminator(BaseModel):
    optimizer_class: Type[th.optim.Adam]

    def __init__(self,
                 lr_schedule: Schedule,
                 net_arch: List[int],
                 activation_fn: Type[nn.Module],
                 **kwargs):
        super().__init__(**kwargs)
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule):
        """Constructs the termination network, consisting of an MLP extractor
        and a final net layer modeling a Bernoulli distribution."""
        layers, _ = get_net_from_layer_dims(
            layers_dims=self.net_arch,
            in_dim=get_flattened_obs_dim(self.observation_space),
            activation_fn=self.activation_fn
        )
        self.mlp_extractor = nn.Sequential(*layers).to(self.device)
        self.bernoulli = BernoulliDistribution(1)
        self.net = self.bernoulli.proba_distribution_net(latent_dim=self.net_arch[-1])
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        """
        Forward pass in terminator network

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: termination and log probability of terminating
        """
        distribution = self.get_distribution(obs)
        termination = distribution.get_actions(deterministic=deterministic).squeeze(0)
        log_prob = distribution.log_prob(termination)
        return termination.type(th.BoolTensor), log_prob

    def get_distribution(self, obs: th.Tensor) -> BernoulliDistribution:
        latent_tn = self._get_latent(obs)
        return self._get_dist_from_latent(latent_tn)

    def _get_latent(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return self.mlp_extractor.forward(features)

    def _get_dist_from_latent(self, latent_tn: th.Tensor) -> BernoulliDistribution:
        mean_termination = self.net(latent_tn)
        return self.bernoulli.proba_distribution(action_logits=mean_termination)

    def evaluate(self, obs: th.Tensor, terminations: th.Tensor) -> (th.Tensor, th.Tensor):
        """
        Evaluate terminations according to the current policy, given the observations.

        :param obs: Observations
        :param terminations: Terminations
        :return: estimated termination log likelihood and entropy
        """
        dist = self.get_distribution(obs)
        return dist.log_prob(terminations), dist.entropy()


class OptionCollection:
    """Rudimentary implementation to handle multiple options at once.
    Efficiency of this code can be increased greatly (especially for high
    number of options)."""

    def __init__(self, options: list[Option]):
        self.options = np.array(options, dtype='object')

    def set_training_mode(self, mode: bool) -> None:
        for option in self.options:
            option.set_training_mode(mode)

    def reset_noise(self, n_envs: int = 1) -> None:
        for option in self.options:
            option.reset_noise(n_envs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> (th.Tensor, th.Tensor, th.Tensor):
        assert len(obs) == len(self.options)
        all_actions = []
        all_values = []
        all_log_probs = []
        for i, option in enumerate(self.options):
            obs_tensor = th.unsqueeze(obs[i], dim=0)
            actions, values, log_prob = option.forward(obs=obs_tensor, deterministic=deterministic)
            all_actions.append(actions)
            all_values.append(values)
            all_log_probs.append(log_prob)
        return th.hstack(all_actions), th.vstack(all_values), th.hstack(all_log_probs)

    def forward_terminator(self, obs: th.Tensor, deterministic: bool = False) -> (th.Tensor, th.Tensor):
        assert len(obs) == len(self.options)
        all_terminations = []
        all_log_probs = []
        for i, option in enumerate(self.options):
            obs_tensor = th.unsqueeze(obs[i], dim=0)
            terminations, log_probs = option.terminator.forward(obs_tensor, deterministic)
            all_log_probs.append(log_probs)
            all_terminations.append(terminations)
        return th.hstack(all_terminations).type(th.BoolTensor), th.hstack(all_log_probs)

    def evaluate_terminations(self, obs: th.Tensor, terminations: th.Tensor) -> (th.Tensor, th.Tensor):
        assert len(obs) == len(self.options)
        all_log_likelihoods = []
        for i, option in enumerate(self.options):
            obs_tensor = th.unsqueeze(obs[i], dim=0)
            termination_dist = option.terminator.get_distribution(obs_tensor)
            log_prob = termination_dist.log_prob(terminations)
            all_log_likelihoods.append(log_prob)
        return th.hstack(all_log_likelihoods)

    def get_termination_dist(self, obs: th.Tensor) -> BernoulliDistribution:
        assert len(self.options) == 1
        return self.options[0].terminator.get_distribution(obs)

    def __getitem__(self, item: Union[int, th.Tensor, np.ndarray]) -> Union[OptionCollection, Option]:
        if isinstance(item, th.Tensor):
            item = item.clone().cpu().numpy()

        if isinstance(item, np.ndarray):
            return OptionCollection(self.options[item])
        elif isinstance(item, int):
            return self.options[item]

    def __len__(self):
        return len(self.options)
