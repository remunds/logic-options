import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution

class MetaPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(MetaPolicy, self).__init__(*args, **kwargs)
    
    # def forward(self, obs, deterministic: bool = False) -> tuple[th.Tensor, Distribution]: 
    #     """
    #     Forward pass in all the networks (actor and critic)

    #     :param obs: Observation
    #     :param deterministic: Not in use 
    #     :return: value and action-distriburtion
    #     """
    #     # Preprocess the observation if needed
    #     features = self.extract_features(obs)
    #     if self.share_features_extractor:
    #         latent_pi, latent_vf = self.mlp_extractor(features)
    #     else:
    #         pi_features, vf_features = features
    #         latent_pi = self.mlp_extractor.forward_actor(pi_features)
    #         latent_vf = self.mlp_extractor.forward_critic(vf_features)
    #     # Evaluate the values for the given observations
    #     values = self.value_net(latent_vf)
    #     distribution = self._get_action_dist_from_latent(latent_pi)
    #     return values, distribution 

    def forward(self, obs, deterministic: bool = False) -> tuple[th.Tensor, Distribution]: 
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Not in use 
        :return: value and action-distriburtion
        """
        dist = self.get_distribution(obs)
        values = self.predict_values(obs)
        return values, dist