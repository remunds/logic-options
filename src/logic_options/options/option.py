from __future__ import annotations

from typing import Type, Union, Optional, Dict, List, Tuple

import numpy as np
import torch as th
from gymnasium.spaces import Space
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.vec_env import VecNormalize
from torch import nn

from logic_options.utils.common import get_net_from_layer_dims
from logic_options.options.meta_policy import MetaPolicy


class Option(nn.Module):
    """Hosts an actor-critic module (policy and value function) and a terminator module.

        :param lr_schedule:
        :param observation_space:
        :param action_space:
        :param policy: Optional: The policy of a pre-trained model to use instead of a
            freshly generated neural policy.
        :param policy_trainable: If False, actor-critic weights don't get updated. Useful when
            this option object is a pre-trained model (submitted via policy).
        :param terminator_trainable: If False, terminator weights don't get updated. Useful when
            using a pre-defined terminator.
        :param vec_norm: VecNormalize env used to normalize the input if a pre-trained
            policy is used (submitted via policy) that requires normalization.
        :param net_arch:
        :param kwargs: Any more parameters for the ActorCritic class.
    """

    def __init__(self,
                 lr_schedule: Schedule,
                 observation_space: Space = None,
                 action_space: Space = None,
                 policy: MetaPolicy = None,
                 policy_trainable: bool = True,
                 value_fn_trainable: bool = True,
                 terminator_trainable: bool = True,
                 vec_norm: VecNormalize = None,
                 terminator: Terminator = None,
                 net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                 **kwargs):
        assert (policy is not None
                or observation_space is not None
                and action_space is not None)

        super().__init__()
        if policy is not None:
            self._policy = policy
        else:
            # self._policy = ActorCriticPolicy(
            self._policy = MetaPolicy(
                observation_space,
                action_space,
                lr_schedule,
                features_extractor_class=FlattenExtractor,
                features_extractor_kwargs=None,
                share_features_extractor=True,
                net_arch=net_arch,
                **kwargs
            )

        if terminator_trainable and not value_fn_trainable:
            print("You are about to train option terminators without policy training. This might "
                  "lead to suboptimal results. It is recommended to train both together.")

        self.vec_norm = vec_norm
        self.normalize_input = self.vec_norm is not None
        self.observation_space = self._policy.observation_space
        self.action_space = self._policy.action_space
        self.policy_trainable = policy_trainable
        self.value_fn_trainable = value_fn_trainable
        self.terminator_trainable = terminator_trainable

        if isinstance(net_arch, dict):
            tn_net_arch = net_arch["tn"]
        elif isinstance(self._policy.net_arch, dict):
            tn_net_arch = self._policy.net_arch["pi"]
        else:
            tn_net_arch = self._policy.net_arch

        if terminator is not None:
            self._terminator = terminator
        else:
            self._terminator = Terminator(
                observation_space=self.observation_space,
                action_space=self.action_space,
                lr_schedule=lr_schedule,
                features_extractor=self._policy.features_extractor,
                net_arch=tn_net_arch,
                activation_fn=self._policy.activation_fn,
                optimizer_class=self._policy.optimizer_class,
                optimizer_kwargs=self._policy.optimizer_kwargs,
                normalize_images=self._policy.normalize_images,
            )
        import ipdb; ipdb.set_trace()

    def get_policy(self):
        return self._policy

    def get_terminator(self):
        return self._terminator

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        return self.forward_actor(obs, deterministic)

    def _normalize(self, obs: th.Tensor):
        if self.normalize_input:
            return th.tensor(self.vec_norm.normalize_obs(np.array(obs.cpu())), device=obs.device)
        return obs

    def forward_actor(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        obs = self._normalize(obs)
        return self._policy.forward(obs, deterministic)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        obs = self._normalize(obs)
        return self._policy.predict_values(obs)

    def forward_terminator(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor]:
        obs = self._normalize(obs)
        return self._terminator.forward(obs, deterministic)

    def get_terminator_dist(self, obs: th.Tensor) -> CategoricalDistribution:
        obs = self._normalize(obs)
        return self._terminator.get_distribution(obs)

    def evaluate_terminations(self, obs: th.Tensor, terminations: th.Tensor) -> (th.Tensor, th.Tensor):
        obs = self._normalize(obs)
        return self._terminator.evaluate(obs, terminations)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        obs = self._normalize(obs)
        return self._policy.evaluate_actions(obs, actions)

    def set_training_mode(self, mode: bool):
        self._policy.set_training_mode(mode)
        self._terminator.set_training_mode(mode)

    def reset_noise(self, n_envs: int = 1):
        self._policy.reset_noise(n_envs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def to(self, device):
        self._policy = self._policy.to(device)
        self._terminator = self._terminator.to(device)
        return self


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
        self.categorical = CategoricalDistribution(2)
        self.net = self.categorical.proba_distribution_net(latent_dim=self.net_arch[-1])
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

    def get_distribution(self, obs: th.Tensor) -> CategoricalDistribution:
        latent_tn = self._get_latent(obs)
        return self._get_dist_from_latent(latent_tn)

    def _get_latent(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return self.mlp_extractor.forward(features)

    def _get_dist_from_latent(self, latent_tn: th.Tensor) -> CategoricalDistribution:
        mean_termination = self.net(latent_tn)
        # starts low at tensor([[-0.6931,  0.6931]], grad_fn=<LogSoftmaxBackward>)
        # but grows with training
        # tensor([[-1.2959,  1.4343]], device='cuda:0')
        if th.isnan(mean_termination).any():
            print("Found NaN in termination logits")
            import ipdb; ipdb.set_trace()
        return self.categorical.proba_distribution(action_logits=mean_termination)

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
            terminations, log_probs = option.forward_terminator(obs_tensor, deterministic)
            all_log_probs.append(log_probs)
            all_terminations.append(terminations)
        return th.hstack(all_terminations).type(th.BoolTensor), th.hstack(all_log_probs)

    def evaluate_terminations(self, obs: th.Tensor, terminations: th.Tensor) -> (th.Tensor, th.Tensor):
        assert len(obs) == len(self.options)
        all_log_likelihoods = []
        for i, option in enumerate(self.options):
            obs_tensor = th.unsqueeze(obs[i], dim=0)
            termination_dist = option.get_terminator_dist(obs_tensor)
            log_prob = termination_dist.log_prob(terminations)
            all_log_likelihoods.append(log_prob)
        return th.hstack(all_log_likelihoods)

    def get_termination_dist(self, obs: th.Tensor) -> CategoricalDistribution:
        assert len(self.options) == 1
        return self.options[0].get_terminator_dist(obs)

    def __getitem__(self, item: Union[int, th.Tensor, np.ndarray]) -> Union[OptionCollection, Option]:
        if isinstance(item, th.Tensor):
            item = item.clone().cpu().numpy()

        if isinstance(item, np.ndarray):
            return OptionCollection(self.options[item])
        elif isinstance(item, int):
            return self.options[item]

    def __len__(self):
        return len(self.options)
