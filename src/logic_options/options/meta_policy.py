import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution

class MetaPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(MetaPolicy, self).__init__(*args, **kwargs)
    
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