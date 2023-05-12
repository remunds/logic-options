from __future__ import annotations

import os.path
from math import exp
from copy import deepcopy
import signal

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli

from experience_replay import ReplayBuffer
from logger import Logger
from utils import to_tensor
from param_schedule import ParamScheduler

MODEL_BASE_PATH = "../out/models/"


class OptionCritic(nn.Module):
    def __init__(self,
                 name: str,
                 env,
                 num_options: int,
                 is_pixel: bool,
                 temperature=1.0,
                 termination_regularization: float = 0.01,
                 entropy_regularization: float = 0.01,
                 device='cpu',
                 testing=False):
        """
        :param name: The agent's name (used for model save path)
        :param env: The Gymnasium environment the agent is going to interact with
        :param num_options: Number of options to create and learn.
        :param is_pixel: Whether the state is a pixel matrix or raw object data.
        :param temperature: Action distribution softmax temperature
        :param termination_regularization: Regularization factor to decrease termination probability =>
            encourage longer options.
        :param entropy_regularization: Regularization factor to increase policy entropy.
        :param device:
        :param testing:
        """

        super(OptionCritic, self).__init__()

        self.name = name

        self.in_shape = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.termination_regularization = termination_regularization
        self.entropy_regularization = entropy_regularization

        self.temperature = temperature
        self.num_steps = 0

        self.latent = self._initialize_latent_model(is_pixel)
        self.latent_dimension = 512 if is_pixel else 64

        self.Q = nn.Linear(self.latent_dimension, num_options)  # Policy-Over-Options
        self.terminations = nn.Linear(self.latent_dimension, num_options)  # Option-Termination
        self.options_W = nn.Parameter(torch.zeros(num_options, self.latent_dimension, self.num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, self.num_actions))

        self.to(device)
        self.train(not testing)

        self.model_dir = get_model_dir(model_name=self.name, env_name=env.spec.id)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.running = True

    def _initialize_latent_model(self, is_pixel):
        if is_pixel:
            return nn.Sequential(
                nn.Conv2d(self.in_shape, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.modules.Flatten(),
                nn.Linear(7 * 7 * 64, 512),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Linear(self.in_shape, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU()
            )

    def _get_latent(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.latent(obs)
        return state

    def practice(self,
                 env,
                 num_transitions: int = 4e6,
                 learning_rate: float = 0.0005,
                 discount_factor: float = 0.99,
                 epsilon: dict = None,
                 replay_buffer_length: int = 10000,
                 batch_size: int = 32,
                 freeze_interval: int = 200,
                 critic_replay_period: int = 4,
                 max_episode_length: int = 10000,
                 seed: int = 0):
        """
        Let agent interact with environment, observe and learn.

        :param env: The initialized Gymnasium environment
        :param num_transitions: Number of total environment steps after which to stop practice.
        :param learning_rate: Alpha
        :param discount_factor: Gamma
        :param epsilon: Epsilon parameters
        :param replay_buffer_length: The maximum number of observations that fit into the experience
            replay buffer.
        :param batch_size: Replay batch size
        :param freeze_interval:
        :param critic_replay_period: Number of transitions before each SGD update (replay)
        :param max_episode_length:
        :param seed:
        :return:
        """
        if epsilon is None:
            epsilon = ParamScheduler(0, "const")
        else:
            epsilon = ParamScheduler(**epsilon)

        # Create a prime network for more stable Q values
        option_critic_prime = deepcopy(self)

        optim = torch.optim.RMSprop(self.parameters(), lr=learning_rate)

        np.random.seed(seed)  # TODO: deprecated
        torch.manual_seed(seed)  # TODO: deprecated

        buffer = ReplayBuffer(capacity=replay_buffer_length, seed=seed)
        logger = Logger(log_dir=self.model_dir + "log")

        transition = 0

        # Setup SIGINT handler
        signal.signal(signal.SIGINT, self.stop_practice)

        # Iterate over episodes
        while transition < num_transitions and self.running:
            ret = 0  # return (= sum of all rewards)
            option_lengths = {opt: [] for opt in range(self.num_options)}

            state, _ = env.reset()
            greedy_option = self.choose_option_greedy(state)
            current_option = 0

            done = False
            episode_length = 0
            terminate_option = True
            current_option_length = 0
            eps = epsilon.get_value(transition)

            # Iterate over transitions
            while not done and episode_length < max_episode_length:
                if terminate_option:
                    option_lengths[current_option].append(current_option_length)
                    current_option = np.random.choice(self.num_options) if np.random.rand() < eps else greedy_option
                    current_option_length = 0

                # Choose action
                action, logp, entropy = self.get_action(state, current_option)

                # Perform transition
                next_state, reward, done, _, _ = env.step(action)

                # Save transition
                buffer.push(state, current_option, reward, next_state, done)
                ret += reward

                # Train
                actor_loss, critic_loss = None, None
                if len(buffer) > batch_size:
                    actor_loss = self.actor_loss(state, current_option, logp, entropy,
                                                 reward, done, next_state, option_critic_prime, discount_factor)
                    loss = actor_loss

                    if transition % critic_replay_period == 0:
                        data_batch = buffer.sample(batch_size)
                        critic_loss = self.critic_loss(option_critic_prime, data_batch, discount_factor)
                        loss += critic_loss

                    # SGD for actor (and critic)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    # Copy parameters from critic model to critic prime
                    if transition % freeze_interval == 0:
                        option_critic_prime.load_state_dict(self.state_dict())

                terminate_option, greedy_option = self.predict_option_termination(next_state, current_option)

                transition += 1
                episode_length += 1
                current_option_length += 1
                state = next_state

                logger.log_data(transition, actor_loss, critic_loss, entropy.item(), eps)

            logger.log_episode(transition, ret, option_lengths, episode_length, eps)

        self.save()

    def play(self, env, max_episode_length: int = np.inf):
        """Let agent interact with environment and render."""

        # Setup SIGINT handler
        signal.signal(signal.SIGINT, self.stop_practice)

        transition = 0
        episode = 0

        print("Playing...")

        # Iterate over episodes
        self.running = True
        while self.running:
            ret = 0  # return (= sum of all rewards)

            state, _ = env.reset()
            greedy_option = self.choose_option_greedy(state)
            current_option = 0

            done = False
            episode_length = 0
            terminate_option = True

            # Iterate over transitions
            while not done and episode_length < max_episode_length:
                if terminate_option:
                    current_option = greedy_option

                # Choose action
                action, logp, entropy = self.get_action(state, current_option)

                # Perform transition
                next_state, reward, done, _, _ = env.step(action)
                ret += reward

                terminate_option, greedy_option = self.predict_option_termination(next_state, current_option)

                transition += 1
                episode_length += 1
                state = next_state

            episode += 1

            print(f"Episode {episode} - Return: {ret} - Length: {episode_length}")

    def get_Q(self, state):
        return self.Q(state)

    def predict_option_termination(self, state, current_option):
        latent = self._get_latent(to_tensor(state))
        termination = self.terminations(latent)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()

        Q = self.get_Q(latent)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()

    def get_terminations(self, state):
        return self.terminations(state).sigmoid()

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
        Q = self.get_Q(latent)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps

    def stop_practice(self, sig, frame):
        print("Stopping practice...")
        self.running = False

    def critic_loss(self, model_prime, data_batch, discount_factor):
        obs, options, rewards, next_obs, dones = data_batch
        batch_idx = torch.arange(len(options)).long()
        options = torch.LongTensor(options).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        masks = 1 - torch.FloatTensor(dones).to(self.device)

        # The loss is the TD loss of Q and the update target, so we need to calculate Q
        states = self._get_latent(to_tensor(obs)).squeeze(0)
        Q = self.get_Q(states)

        # the update target contains Q_next, but for stable learning we use prime network for this
        next_states_prime = model_prime._get_latent(to_tensor(next_obs)).squeeze(0)
        next_Q_prime = model_prime.get_Q(next_states_prime)  # detach?

        # Additionally, we need the beta probabilities of the next state
        next_states = self._get_latent(to_tensor(next_obs)).squeeze(0)
        next_termination_probs = self.get_terminations(next_states).detach()
        next_options_term_prob = next_termination_probs[batch_idx, options]

        # Now we can calculate the update target gt
        gt = rewards + masks * discount_factor * \
             ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + next_options_term_prob *
              next_Q_prime.max(dim=-1)[0])

        # to update Q we want to use the actual network, not the prime
        td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
        return td_err

    def actor_loss(self, obs, option, logp, entropy, reward, done, next_obs, model_prime, discount_factor):
        state = self._get_latent(to_tensor(obs))
        next_state = self._get_latent(to_tensor(next_obs))
        next_state_prime = model_prime._get_latent(to_tensor(next_obs))

        option_term_prob = self.get_terminations(state)[:, option]
        next_option_term_prob = self.get_terminations(next_state)[:, option].detach()

        Q = self.get_Q(state).detach().squeeze()
        next_Q_prime = model_prime.get_Q(next_state_prime).detach().squeeze()

        # Target update gt
        gt = reward + (1 - done) * discount_factor * \
             ((1 - next_option_term_prob) * next_Q_prime[option] + next_option_term_prob * next_Q_prime.max(dim=-1)[0])

        # The termination loss
        termination_loss = option_term_prob * (Q[option].detach() - Q.max(dim=-1)[0].detach() +
                                               self.termination_regularization) * (1 - done)

        # actor-critic policy gradient with entropy regularization
        policy_loss = -logp * (gt.detach() - Q[option]) - self.entropy_regularization * entropy
        actor_loss = termination_loss + policy_loss
        return actor_loss

    def save(self):
        torch.save(self, self.model_dir + "result.pckl")

    @staticmethod
    def load(env_name, model_name) -> OptionCritic:
        model_path = get_model_dir(env_name, model_name) + "result.pckl"
        return torch.load(model_path)


def get_model_dir(env_name, model_name):
    return MODEL_BASE_PATH + f"{env_name}/{model_name}/"