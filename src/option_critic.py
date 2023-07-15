from __future__ import annotations

from typing import Union

from gymnasium import Env
from ocatari.core import OCAtari
import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli

from utils import to_tensor, categorize_objects_into_dict, get_category_counts

MODEL_BASE_PATH = "out/models/"
MODEL_BASE_FILENAME = "model"
CHECKPOINT_SAVE_PERIOD = 1000000
SUMMARY_WRITE_INTERVAL = 100  # no. of transitions


class OptionCritic(nn.Module):
    def __init__(self,
                 env: Union[Env, OCAtari],
                 num_options: int,
                 is_object_centric: bool,
                 latent_dimension: int,
                 temperature: float = 1.0,
                 termination_regularization: float = 0.01,
                 entropy_regularization: float = 0.01,
                 device='cpu',
                 testing=False):
        """
        :param env: The Gymnasium environment the agent is going to interact with
        :param num_options: Number of options to create and learn.
        :param is_object_centric: If true, the state representation is object-centric (positions and
            velocities), otherwise a pixel matrix.
        :param temperature: Action distribution softmax temperature. Increase temperature to
            even out action selection probability. Set to 1 to deactivate.
        :param termination_regularization: Regularization factor. If increased, option termination
            probability gets decreased, i.e., longer options are encouraged. Set to 0 do disable this
            regularization.
        :param entropy_regularization: Regularization factor to control policy entropy. Increase this
            factor to worsen policy loss, enforcing higher policy entropy. Setting to zero deactivates
            this regularizer.
        :param device:
        :param testing:
        """

        assert num_options > 0

        super(OptionCritic, self).__init__()

        if is_object_centric:
            self.max_num_objects = len(env.max_objects)
            self.in_shape = self.max_num_objects * 4
        else:
            self.in_shape = env.observation_space.shape[0]

        if is_object_centric:
            objects_categorized = categorize_objects_into_dict(env.max_objects)
            self.max_object_counts = get_category_counts(objects_categorized)

        self.num_actions = env.action_space.n
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.termination_regularization = termination_regularization
        self.entropy_regularization = entropy_regularization

        self.temperature = temperature
        self.num_steps = 0

        self.latent_dimension = latent_dimension
        self.latent = self._initialize_latent_model(is_object_centric)

        self.option_values = nn.Linear(self.latent_dimension, num_options)  # inter-option policy
        self.termination_probabilities = nn.Linear(self.latent_dimension, num_options)
        self.options_W = nn.Parameter(torch.zeros(num_options, self.latent_dimension, self.num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, self.num_actions))

        self.to(device)
        self.train(not testing)

    def _initialize_latent_model(self, is_object_centric):
        if is_object_centric:
            return nn.Sequential(
                nn.Linear(self.in_shape, self.latent_dimension // 2),
                nn.ReLU(),
                nn.Linear(self.latent_dimension // 2, self.latent_dimension),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(self.in_shape, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.modules.Flatten(),
                nn.Linear(7 * 7 * 64, self.latent_dimension),
                nn.ReLU()
            )

    def _get_latent(self, obs: torch.Tensor):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.latent(obs)
        return state

    def get_option_values(self, state):
        return self.option_values(state)

    def predict_option_termination(self, state, current_option):
        latent = self._get_latent(to_tensor(state))
        termination = self.termination_probabilities(latent)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()

        option_values = self.get_option_values(latent)
        next_option = option_values.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()

    def get_termination_probabilities(self, latent: torch.Tensor):
        return self.termination_probabilities(latent).sigmoid()

    def get_action(self, state, option):
        """Given an environment state, samples an action of the specified option policy."""
        latent = self._get_latent(to_tensor(state))

        logits = latent.data @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy

    def choose_option_greedy(self, state):
        latent = self._get_latent(to_tensor(state))
        Q = self.get_option_values(latent)
        return Q.argmax(dim=-1).item()

    def critic_loss(self, model_prime, data_batch, discount_factor):
        states, options, rewards, next_states, dones = data_batch

        batch_ids = torch.arange(len(options)).long()
        options = torch.LongTensor(options).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        masks = 1 - torch.FloatTensor(dones).to(self.device)

        # The loss is the TD loss of Q and the update target, so we need to calculate Q
        latents = self._get_latent(to_tensor(states)).squeeze(0)
        option_values = self.get_option_values(latents)

        # The update target contains Q_next, but for stable learning we use prime network for this
        next_latents_prime = model_prime._get_latent(to_tensor(next_states)).squeeze(0)
        next_option_values_prime = model_prime.get_option_values(next_latents_prime)  # TODO: detach?

        # Additionally, we need beta (the termination probabilities) of the next state
        next_latents = self._get_latent(to_tensor(next_states)).squeeze(0)
        next_termination_probs = self.get_termination_probabilities(next_latents).detach()
        next_option_termination_probs = next_termination_probs[batch_ids, options]

        # Now we can calculate the update target g_t
        g_t = rewards + masks * discount_factor * (
                (1 - next_option_termination_probs) * next_option_values_prime[batch_ids, options]
                + next_option_termination_probs * next_option_values_prime.max(dim=-1)[0]
        )

        # To update Q we want to use the actual network, not the prime
        td_err = (option_values[batch_ids, options] - g_t.detach()).pow(2).mul(0.5).mean()
        return td_err

    def actor_loss(self, state, option, logp, entropy, reward, done, next_state, model_prime, discount_factor):
        # Compute latent vectors
        latent = self._get_latent(to_tensor(state))
        next_latent = self._get_latent(to_tensor(next_state))
        next_latent_prime = model_prime._get_latent(to_tensor(next_state))

        # Compute termination probabilities of current option for current state and for next state
        termination_probability = self.get_termination_probabilities(latent)[:, option]
        next_termination_probability = self.get_termination_probabilities(next_latent)[:, option].detach()

        # Compute Q-values for options
        option_values = self.get_option_values(latent).detach().squeeze(0)
        next_option_values_prime = model_prime.get_option_values(next_latent_prime).detach().squeeze(0)

        # One-step off-policy update target
        if done:
            g_t = reward
        else:
            g_t = reward + discount_factor * (
                    (1 - next_termination_probability) * next_option_values_prime[option]
                    + next_termination_probability * next_option_values_prime.max(dim=-1)[0]
            ).detach()

        # Compute termination loss
        if done:
            termination_loss = 0
        else:
            option_advantage = (option_values[option] - option_values.max(dim=-1)[0]).detach()
            termination_loss = termination_probability * (option_advantage + self.termination_regularization)

        # Actor-critic policy gradient with entropy regularization
        policy_loss = -logp * (g_t - option_values[option]) - self.entropy_regularization * entropy
        actor_loss = termination_loss + policy_loss
        return actor_loss
