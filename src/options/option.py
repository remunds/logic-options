from __future__ import annotations

from typing import Type, Union, Optional

import numpy as np
import torch as th
from gymnasium.spaces import Space
from stable_baselines3.common.distributions import BernoulliDistribution
from stable_baselines3.common.policies import ActorCriticPolicy, BaseModel
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import FlattenExtractor
from torch import nn
# from torchviz import make_dot

from utils.common import get_net_from_layer_dims


class Option(ActorCriticPolicy):
    """AC model extended by a flow for termination prediction."""

    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 net_arch: Optional[Union[list[int], dict[str, list[int]]]],
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 **kwargs):
        super().__init__(observation_space,
                         action_space,
                         net_arch=net_arch,
                         activation_fn=activation_fn,
                         features_extractor_class=FlattenExtractor,
                         features_extractor_kwargs=None,
                         share_features_extractor=True,
                         **kwargs)

        self._build_tn_net(net_arch)

        # Create and plot computation graph (for verification)
        # x = th.randn(1, *self.observation_space.shape)
        # y = (*self(x), self.predict_termination(x))
        # make_dot(y,
        #          params=dict(self.named_parameters()),
        #          show_attrs=True)

    def _build_tn_net(self, net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None):
        """Constructs the termination network, consisting of an MLP extractor
        and a final net layer modeling a Bernoulli distribution."""
        layers_dims = net_arch["tn"] if isinstance(net_arch, dict) else net_arch
        termination_layers, self.latent_dim_tn = \
            get_net_from_layer_dims(layers_dims=layers_dims,
                                    in_dim=get_flattened_obs_dim(self.observation_space),
                                    activation_fn=self.activation_fn)
        self.termination_mlp_extractor = nn.Sequential(*termination_layers).to(self.device)
        self.termination_dist = BernoulliDistribution(1)
        self.terminator_net = self.termination_dist.proba_distribution_net(latent_dim=self.mlp_extractor.latent_dim_pi)

    def _get_tn_latent(self, obs: th.Tensor) -> th.Tensor:
        features = BaseModel.extract_features(self, obs, self.features_extractor)
        return self.termination_mlp_extractor.forward(features)

    def _get_termination_dist_from_latent(self, latent_tn: th.Tensor) -> BernoulliDistribution:
        mean_termination = self.terminator_net(latent_tn)
        return self.termination_dist.proba_distribution(action_logits=mean_termination)

    def get_termination_dist(self, obs: th.Tensor) -> BernoulliDistribution:
        latent_tn = self._get_tn_latent(obs)
        return self._get_termination_dist_from_latent(latent_tn)

    def predict_termination(self, obs: th.Tensor) -> th.Tensor:
        latent_tn = self._get_tn_latent(obs)
        return self.terminator_net(latent_tn)

    def forward_terminator(self, obs: th.Tensor, deterministic: bool = False):
        """
        Forward pass in terminator network

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: termination and log probability of that termination
        """
        distribution = self.get_termination_dist(obs)
        termination = distribution.get_actions(deterministic=deterministic).squeeze(0)
        log_prob = distribution.log_prob(termination)
        return termination.type(th.BoolTensor), log_prob

    def evaluate_terminations(self, obs: th.Tensor, terminations: th.Tensor) -> (th.Tensor, th.Tensor):
        """
        Evaluate terminations according to the current policy, given the observations.

        :param obs: Observations
        :param terminations: Terminations
        :return: estimated termination log likelihood
        """
        termination_dist = self.get_termination_dist(obs)
        return termination_dist.log_prob(terminations), termination_dist.entropy()


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
            terminations, log_probs = option.forward_terminator(obs_tensor, deterministic)
            all_log_probs.append(log_probs)
            all_terminations.append(terminations)
        return th.hstack(all_terminations).type(th.BoolTensor), th.hstack(all_log_probs)

    def evaluate_terminations(self, obs: th.Tensor, terminations: th.Tensor) -> (th.Tensor, th.Tensor):
        assert len(obs) == len(self.options)
        all_log_likelihoods = []
        for i, option in enumerate(self.options):
            obs_tensor = th.unsqueeze(obs[i], dim=0)
            termination_dist = option.get_termination_dist(obs_tensor)
            log_prob = termination_dist.log_prob(terminations)
            all_log_likelihoods.append(log_prob)
        return th.hstack(all_log_likelihoods)

    def get_termination_dist(self, obs: th.Tensor) -> BernoulliDistribution:
        assert len(self.options) == 1
        return self.options[0].get_termination_dist(obs)

    def __getitem__(self, item: Union[int, th.Tensor, np.ndarray]) -> Union[OptionCollection, Option]:
        if isinstance(item, th.Tensor):
            item = item.clone().cpu().numpy()

        if isinstance(item, np.ndarray):
            return OptionCollection(self.options[item])
        elif isinstance(item, int):
            return self.options[item]

    def __len__(self):
        return len(self.options)
