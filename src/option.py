from __future__ import annotations

from typing import Type, Union, Optional

from torchviz import make_dot
import numpy as np
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy, BaseModel
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.distributions import BernoulliDistribution
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from gymnasium.spaces import Space


class Option(ActorCriticPolicy):
    """AC model extended by a flow for termination prediction."""

    def __init__(self,
                 observation_space: Space,
                 device: th.device,
                 net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 **kwargs):
        super().__init__(observation_space=observation_space,
                         net_arch=net_arch,  # TODO: adjust this parameter
                         activation_fn=activation_fn,
                         features_extractor_class=FlattenExtractor,
                         features_extractor_kwargs=None,
                         share_features_extractor=True,
                         **kwargs)

        # Build termination network (MLP extraction)
        layers_dims = net_arch["tn"] if isinstance(net_arch, dict) else net_arch
        termination_layers, self.latent_dim_tn = \
            get_net_from_layer_dims(layers_dims=layers_dims,
                                    in_dim=get_flattened_obs_dim(observation_space),
                                    activation_fn=activation_fn)
        self.termination_mlp_extractor = nn.Sequential(*termination_layers).to(device)

        self.termination_dist = BernoulliDistribution(1)
        self.terminator_net = self.termination_dist.proba_distribution_net(latent_dim=self.mlp_extractor.latent_dim_pi)

        # Create and plot computation graph (for verification)
        x = th.randn(1, *self.observation_space.shape)
        y = (*self(x), self.predict_termination(x))
        make_dot(y,
                 params=dict(self.named_parameters()),
                 show_attrs=True)

    def get_tn_latent(self, obs: th.Tensor):
        features = BaseModel.extract_features(self, obs, self.features_extractor)
        return self.termination_mlp_extractor.forward(features)

    def get_termination_dist_from_latent(self, latent_tn: th.Tensor):
        mean_termination = self.terminator_net(latent_tn)
        return self.termination_dist.proba_distribution(action_logits=mean_termination)

    def get_termination_dist(self, obs: th.Tensor):
        latent_tn = self.get_tn_latent(obs)
        return self.get_termination_dist_from_latent(latent_tn)

    def predict_termination(self, obs: th.Tensor) -> th.Tensor:
        latent_tn = self.get_tn_latent(obs)
        return self.terminator_net(latent_tn)


# class Option(nn.Module):
#     def __init__(self, feature_dim: int,
#                  net_arch: Union[list[int], dict[str, list[int]]] = None,
#                  activation_fn: Type[nn.Module] = th.nn.Tanh,
#                  device: Union[th.device, str] = "auto"):
#         """
#         :param feature_dim:
#         :param net_arch: The layer dimensions to be used for the MLP latent extractor
#             network. If unspecified, there won't be an MLP extractor, i.e., actor, critic,
#             and terminator all use the OC representation vector directly.
#         :param activation_fn:
#         :param device:
#         """
#         super().__init__()
#
#         self.feature_dim = feature_dim
#         device = get_device(device)
#
#         if net_arch is None:
#             net_arch = []
#         self._build_net(net_arch, activation_fn, device)
#
#     def _build_net(self, net_arch, activation_fn, device):
#         # Construct networks
#         if isinstance(net_arch, dict):
#             # Note: if key is not specified, assume linear network
#             pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
#             vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
#             tn_layers_dims = net_arch.get("tn", [])  # Layer sizes of the termination network
#         else:
#             pi_layers_dims = vf_layers_dims = tn_layers_dims = net_arch
#
#         policy_net, last_layer_dim_pi = \
#             get_net_from_layer_dims(pi_layers_dims, in_dim=self.feature_dim, activation_fn=activation_fn)
#         value_net, last_layer_dim_vf = \
#             get_net_from_layer_dims(vf_layers_dims, in_dim=self.feature_dim, activation_fn=activation_fn)
#         termination_net, last_layer_dim_tn = \
#             get_net_from_layer_dims(tn_layers_dims, in_dim=self.feature_dim, activation_fn=activation_fn)
#
#         # Save dim, used to create the distributions
#         self.latent_dim_pi = last_layer_dim_pi
#         self.latent_dim_vf = last_layer_dim_vf
#         self.latent_dim_tn = last_layer_dim_tn
#
#         self.policy_net = nn.Sequential(*policy_net).to(device)  # the actor: intra-option policy pi_omega
#         self.value_net = nn.Sequential(*value_net).to(device)  # the critic: estimates Q_omega(a | s)
#         self.termination_net = nn.Sequential(*termination_net).to(device)  # the terminator: beta


def get_net_from_layer_dims(layers_dims: list[int],
                            in_dim: int,
                            activation_fn: Type[nn.Module]) -> (list[nn.Module], int):
    net: list[nn.Module] = []
    last_layer_dim = in_dim
    for layer_dim in layers_dims:
        net.append(nn.Linear(last_layer_dim, layer_dim))
        net.append(activation_fn())
        last_layer_dim = layer_dim
    return net, last_layer_dim


class OptionCollection:
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

    # def get_termination_dist(self, obs: th.Tensor):
    #     assert len(obs) == len(self.options)
    #     # all_dists = []
    #     termination_probs = []
    #     for i, option in enumerate(self.options):
    #         obs_tensor = th.unsqueeze(obs[i], dim=0)
    #         termination_prob = option.predict_termination(obs_tensor)
    #         termination_probs.append(termination_prob)
    #         # latent_tn = option.get_tn_latent(obs_tensor)
    #         # dist = option.get_termination_dist_from_latent(latent_tn)
    #         # all_dists.append(dist)
    #     termination_dist = BernoulliDistribution
    #     return all_dists

    def forward_terminator(self, obs: th.Tensor, deterministic: bool = False) -> (th.Tensor, th.Tensor):
        assert len(obs) == len(self.options)
        all_terminations = []
        log_probs = []
        for i, option in enumerate(self.options):
            obs_tensor = th.unsqueeze(obs[i], dim=0)
            # latent_tn = option.get_tn_latent(obs_tensor)
            termination_dist = option.get_termination_dist(obs_tensor)
            terminations = termination_dist.get_actions(deterministic=deterministic).squeeze(0)
            log_prob = termination_dist.log_prob(terminations)
            log_probs.append(log_prob)
            # termination = option.terminator_net(latent_tn)
            all_terminations.append(terminations)
        return th.hstack(all_terminations).type(th.BoolTensor), th.hstack(log_probs)

    def __getitem__(self, item: Union[int, th.Tensor, np.ndarray]) -> Union[OptionCollection, Option]:
        if isinstance(item, th.Tensor):
            item = item.clone().cpu().numpy()

        if isinstance(item, np.ndarray):
            return OptionCollection(self.options[item])
        elif isinstance(item, int):
            return self.options[item]
