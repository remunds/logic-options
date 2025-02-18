import importlib
import sys
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import CategoricalDistribution

class FunctionMetaPolicy:
    def __init__(self, function_path, num_options, device):
        # load function
        self.num_options = num_options
        spec = importlib.util.spec_from_file_location("meta_policy", function_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["meta_policy"] = module
        spec.loader.exec_module(module)
        self.meta_policy_func = module.meta_policy
        self.action_space = spaces.Discrete(num_options)
        #TODO: multiple envs
        self.dist = CategoricalDistribution(num_options)
        self.device = device

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, obs, deterministic): 
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :return: None (values), action-distribution
        """
        option_choices = self.meta_policy_func(obs)
        #TODO: multiple envs
        probs = th.zeros((obs.shape[0], self.num_options)).to(self.device)
        #probs shape: (1, 3)
        # so indices to set to 1 has shape: (1,) 
        row_idx = th.arange(len(option_choices))
        probs[row_idx, option_choices] = 1
        dist = self.dist.proba_distribution(probs.log())
        # create values tensor with zeros
        values = th.zeros(obs.shape[0])
        return values, dist

    def set_training_mode(self, training):
        pass

    def reset_noise(self, n_envs):
        pass

    def predict_values(self, obs):
        return th.zeros(obs.shape[0])
    
    def to(self, device):
        return self

    def state_dict(self):
        return {}
    
    def load_state_dict(self, state_dict):
        pass