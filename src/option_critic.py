from __future__ import annotations

import os.path
from math import exp
from copy import deepcopy
import signal
from typing import Union

from gymnasium import Env
import numpy as np
from ocatari.core import OCAtari
import pygame
import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli

from experience_replay import ReplayBuffer
from logger import Logger
from utils import to_tensor, objects_to_numeric_vector, categorize_objects_into_dict, get_category_counts, pad_object_list
from param_schedule import ParamScheduler

MODEL_BASE_PATH = "out/models/"


class OptionCritic(nn.Module):
    def __init__(self,
                 name: str,
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
        :param name: The agent's name (used for model save path)
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

        self.name = name
        self.object_centric = is_object_centric

        env_name = env.spec.name if isinstance(env, Env) else env.game_name
        self.model_dir = get_model_dir(model_name=self.name, env_name=env_name)
        if os.path.exists(self.model_dir):
            ans = input(f"There already exists a model at '{self.model_dir}'. Override that model? (y/n)")
            if ans == 'y':
                os.rmdir(self.model_dir)  # TODO: enable run continuation
            else:
                quit()
        os.makedirs(self.model_dir)

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

        self.running = True

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

    def get_latent(self, obs: torch.Tensor):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.latent(obs)
        return state

    def practice(self,
                 env,
                 num_transitions: int = 4e6,
                 learning_rate: float = 0.0005,
                 optimizer: str = 'Adam',
                 discount_factor: float = 0.99,
                 epsilon: dict = None,
                 replay_buffer_length: int = 10000,
                 replay_start_size: int = 500,
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
        :param replay_start_size:
        :param optimizer:
        :return:
        """

        # initialize_tensorboard()

        if epsilon is None:
            epsilon = ParamScheduler(0, "const")
        else:
            epsilon = ParamScheduler(**epsilon)

        replay_start_size = max(replay_start_size, batch_size)

        # Create a prime network for more stable Q values
        option_critic_prime = deepcopy(self)

        if optimizer == "Adam":
            optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == "RMSprop":
            optim = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Invalid optimizer '{optimizer}' specified.")

        np.random.seed(seed)  # TODO: deprecated
        torch.manual_seed(seed)  # TODO: deprecated

        buffer = ReplayBuffer(capacity=replay_buffer_length, seed=seed)
        logger = Logger(log_dir=self.model_dir + "log")

        transition = 0

        # Setup SIGINT handler
        signal.signal(signal.SIGINT, self.stop_running)

        # Iterate over episodes
        while transition < num_transitions and self.running:
            ret = 0  # return (= sum of all rewards)
            option_lengths = {opt: [] for opt in range(self.num_options)}

            state, _ = env.reset()
            if self.object_centric:
                state = objects_to_numeric_vector(pad_object_list(env.objects, self.max_object_counts))

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
                if self.object_centric:
                    next_state = objects_to_numeric_vector(pad_object_list(env.objects, self.max_object_counts))

                # Save transition
                buffer.push(state, current_option, reward, next_state, done)
                ret += reward

                # Train
                actor_loss, critic_loss = None, None
                if len(buffer) > replay_start_size:
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
        print("Saved model.")

    def play(self, env, max_episode_length: int = np.inf):
        """Let agent interact with environment and render."""

        # Setup SIGINT handler
        signal.signal(signal.SIGINT, self.stop_running)

        transition = 0
        episode = 0

        # Initialize GUI
        rgb_array = env.render()
        pygame.init()
        pygame.display.set_caption(env.spec.name)
        screen = pygame.display.set_mode((rgb_array.shape[1], rgb_array.shape[0]), flags=pygame.SCALED)
        self._render_rgb(screen, rgb_array)
        clock = pygame.time.Clock()

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
            while not done and episode_length < max_episode_length and self.running:
                if terminate_option:
                    current_option = greedy_option

                # Choose action
                action, logp, entropy = self.get_action(state, current_option)

                # Perform transition
                next_state, reward, done, _, _ = env.step(action)
                ret += reward

                # Render environment for human
                rgb_array = env.render()
                clock.tick(30)  # reduce FPS for Atari games
                self._render_rgb(screen, rgb_array, option_id=current_option)

                terminate_option, greedy_option = self.predict_option_termination(next_state, current_option)

                # Finalize transition
                transition += 1
                episode_length += 1
                state = next_state

            episode += 1

            print(f"Episode {episode} - Return: {ret} - Length: {episode_length}")

        pygame.quit()

    def _render_rgb(self, screen, rgb_array, option_id=None):
        # Render RGB image
        rgb_array = np.transpose(rgb_array, (1, 0, 2))
        pygame.pixelcopy.array_to_surface(screen, rgb_array)

        # If given, render option ID in top right corner
        if option_id is not None:
            font = pygame.font.SysFont('Calibri', 16)
            text = font.render("Option " + str(option_id), True, (255, 255, 50), None)
            rect = text.get_rect()
            rect.bottomright = (screen.get_size()[0], screen.get_size()[1])
            pygame.draw.rect(screen, color=(20, 20, 20), rect=rect)
            screen.blit(text, rect)

        pygame.display.flip()
        pygame.event.pump()

    def get_option_values(self, state):
        return self.option_values(state)

    def predict_option_termination(self, state, current_option):
        latent = self.get_latent(to_tensor(state))
        termination = self.termination_probabilities(latent)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()

        option_values = self.get_option_values(latent)
        next_option = option_values.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()

    def get_termination_probabilities(self, latent: torch.Tensor):
        return self.termination_probabilities(latent).sigmoid()

    def get_action(self, state, option):
        """Given an environment state, samples an action of the specified option policy."""
        latent = self.get_latent(to_tensor(state))

        logits = latent.data @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy

    def choose_option_greedy(self, state):
        latent = self.get_latent(to_tensor(state))
        Q = self.get_option_values(latent)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps

    def stop_running(self, sig, frame):
        print("Stopping running...")
        self.running = False

    def critic_loss(self, model_prime, data_batch, discount_factor):
        states, options, rewards, next_states, dones = data_batch

        batch_ids = torch.arange(len(options)).long()
        options = torch.LongTensor(options).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        masks = 1 - torch.FloatTensor(dones).to(self.device)

        # The loss is the TD loss of Q and the update target, so we need to calculate Q
        latents = self.get_latent(to_tensor(states)).squeeze(0)
        option_values = self.get_option_values(latents)

        # The update target contains Q_next, but for stable learning we use prime network for this
        next_latents_prime = model_prime.get_latent(to_tensor(next_states)).squeeze(0)
        next_option_values_prime = model_prime.get_option_values(next_latents_prime)  # TODO: detach?

        # Additionally, we need beta (the termination probabilities) of the next state
        next_latents = self.get_latent(to_tensor(next_states)).squeeze(0)
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
        latent = self.get_latent(to_tensor(state))
        next_latent = self.get_latent(to_tensor(next_state))
        next_latent_prime = model_prime.get_latent(to_tensor(next_state))

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

    def save(self):
        torch.save(self, self.model_dir + "result.pckl")

    @staticmethod
    def load(env_name, model_name) -> OptionCritic:
        model_path = get_model_dir(env_name, model_name) + "result.pckl"
        return torch.load(model_path)


def get_model_dir(env_name, model_name):
    return MODEL_BASE_PATH + f"{env_name}/{model_name}/"


