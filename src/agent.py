from __future__ import annotations

import os.path
import glob
from copy import deepcopy
import signal
from typing import Union

from gymnasium import Env
import numpy as np
from ocatari.core import OCAtari
import pygame
import torch

from experience_replay import ReplayBuffer
from logger import Logger
from utils import objects_to_matrix, categorize_objects_into_dict, get_category_counts, \
    pad_object_list, get_env_name, normalize_object_matrix
from param_schedule import ParamScheduler

MODEL_BASE_PATH = "out/models/"
MODEL_BASE_FILENAME = "model"
CHECKPOINT_SAVE_PERIOD = 1000000
SUMMARY_WRITE_INTERVAL = 100  # no. of transitions


class Agent:
    def __init__(self,
                 name: str,
                 model,
                 env: Union[Env, OCAtari],
                 is_object_centric: bool):
        """
        :param name: The agent's name (used for model save path)
        :param env: The Gymnasium environment the agent is going to interact with
        :param is_object_centric: If true, the state representation is object-centric (positions and
            velocities), otherwise a pixel matrix.
        """

        self.name = name
        self.object_centric = is_object_centric

        self.model = model

        env_name = get_env_name(env)
        self.model_dir = get_model_dir(model_name=self.name, env_name=env_name)

        # Handle any pre-existing model
        if os.path.exists(self.model_dir):
            raise ValueError(f"There already exists a model at {self.model_dir}.")
        else:
            os.makedirs(self.model_dir)

        if is_object_centric:
            objects_categorized = categorize_objects_into_dict(env.max_objects)
            self.max_object_counts = get_category_counts(objects_categorized)

        self.num_steps = 0
        self.running = True
        self.transitions_total = 0
        self.return_record = -np.inf

    def practice(self,
                 env: Union[Env, OCAtari],
                 num_transitions: int = 4e6,
                 learning_rate: float = 0.0005,
                 optimizer: str = 'Adam',
                 discount_factor: float = 0.99,
                 epsilon: dict = None,
                 replay_buffer_length: int = None,
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

        if epsilon is None:
            epsilon = ParamScheduler(0, "const")
        else:
            epsilon = ParamScheduler(**epsilon)

        if replay_buffer_length is None:
            replay_buffer_length = batch_size

        assert replay_buffer_length >= batch_size

        replay_start_size = max(replay_start_size, batch_size)

        # Create a prime network for more stable Q values
        model_prime = deepcopy(self.model)

        if optimizer == "Adam":
            optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == "RMSprop":
            optim = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Invalid optimizer '{optimizer}' specified.")

        np.random.seed(seed)  # TODO: deprecated
        torch.manual_seed(seed)  # TODO: deprecated

        buffer = ReplayBuffer(capacity=replay_buffer_length, seed=seed)
        logger = Logger(log_dir=self.model_dir + "/log")

        transition = 0

        # Setup SIGINT handler
        signal.signal(signal.SIGINT, self.stop_running)

        # Iterate over episodes
        while transition < num_transitions and self.running:
            ret = 0  # return (= sum of all rewards)
            option_lengths = {opt: [] for opt in range(self.model.num_options)}

            state, _ = env.reset()
            if self.object_centric:
                state = self.get_current_oc_env_state(env)

            greedy_option = self.model.choose_option_greedy(state)
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
                    current_option = np.random.choice(self.model.num_options) if np.random.rand() < eps else greedy_option
                    current_option_length = 0

                # Choose action
                action, logp, entropy = self.model.get_action(state, current_option)

                # Perform transition
                next_state, reward, done, _, _ = env.step(action)
                if self.object_centric:
                    next_state = self.get_current_oc_env_state(env)

                # Reward shaping for Pong
                # if "Pong" in env.game_name:
                #     ball_position_y = env.objects[1].y
                #     player_position_y = env.objects[0].y
                #     reward += 0.05 / (1 + abs(ball_position_y - player_position_y))

                # Save transition
                buffer.push(state, current_option, reward, next_state, done)
                ret += reward

                # Train
                actor_loss, critic_loss = None, None
                if len(buffer) > replay_start_size:
                    actor_loss = self.model.actor_loss(state, current_option, logp, entropy,
                                                 reward, done, next_state, model_prime, discount_factor)
                    loss = actor_loss

                    if transition % critic_replay_period == 0:
                        data_batch = buffer.sample(batch_size)
                        critic_loss = self.model.critic_loss(model_prime, data_batch, discount_factor)
                        loss += critic_loss

                    # SGD for actor (and critic)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    # Copy parameters from critic model to critic prime
                    if transition % freeze_interval == 0:
                        model_prime.load_state_dict(self.model.state_dict())

                terminate_option, greedy_option = self.model.predict_option_termination(next_state, current_option)

                self.transitions_total += 1
                transition += 1
                episode_length += 1
                current_option_length += 1
                state = next_state

                if self.transitions_total % SUMMARY_WRITE_INTERVAL == 0:
                    logger.log_data(self.transitions_total, actor_loss, critic_loss, entropy.item(), eps)

                if self.transitions_total % CHECKPOINT_SAVE_PERIOD == 0:
                    self.save()

            logger.log_episode(self.transitions_total, ret, option_lengths, episode_length, eps)

            if ret > self.return_record:
                if self.return_record > -np.inf:
                    logger.logger.info(f"New record: {ret}")
                    self.save(suffix="_record_%.0f" % ret)
                self.return_record = ret

        self.save()

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
            if self.object_centric:
                state = self.get_current_oc_env_state(env)
            greedy_option = self.model.choose_option_greedy(state)
            current_option = 0

            done = False
            episode_length = 0
            terminate_option = True

            # Iterate over transitions
            while not done and episode_length < max_episode_length and self.running:
                if terminate_option:
                    current_option = greedy_option

                # Choose action
                action, logp, entropy = self.model.get_action(state, current_option)

                # Perform transition
                next_state, reward, done, _, _ = env.step(action)
                if self.object_centric:
                    next_state = self.get_current_oc_env_state(env)
                ret += reward

                # Render environment for human
                rgb_array = env.render()
                clock.tick(15)  # reduce FPS for Atari games
                self._render_rgb(screen, rgb_array, option_id=current_option,
                                 object_centric=self.object_centric, env=env)

                terminate_option, greedy_option = self.model.predict_option_termination(next_state, current_option)

                # Finalize transition
                transition += 1
                episode_length += 1
                state = next_state

            episode += 1

            print(f"Episode {episode} - Return: {ret} - Length: {episode_length}")

        pygame.quit()

    def _render_rgb(self, screen, rgb_array, option_id=None, object_centric=False, env=None):
        """Displays an RGB pixel image using pygame.
        Optional: Show the currently used option_id and/or the detected objects
        and their velocities."""
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

        # If given, render object coordinates and velocity vectors
        if object_centric and env is not None:
            padded_object_list = pad_object_list(env.objects, self.max_object_counts)
            object_matrix = objects_to_matrix(padded_object_list)

            for x, y, dx, dy in object_matrix:
                if x == np.nan:
                    continue

                # Draw an 'X' at object center
                pygame.draw.line(screen, color=(255, 255, 255),
                                 start_pos=(x-2, y-2), end_pos=(x+2, y+2))
                pygame.draw.line(screen, color=(255, 255, 255),
                                 start_pos=(x-2, y+2), end_pos=(x+2, y-2))

                # Draw velocity vector
                if dx != 0 or dy != 0:
                    if abs(dx) > 10 or abs(dy) > 10:
                        print(f"Large velocity dx={dx}, dy={dy} encountered!")
                    # TODO: make this an actual arrow
                    pygame.draw.line(screen, color=(100, 200, 255),
                                     start_pos=(float(x), float(y)), end_pos=(x+8*dx, y+8*dy))

        pygame.display.flip()
        pygame.event.pump()

    def stop_running(self, sig, frame):
        print("Stopping running...")
        self.running = False

    def save(self, suffix=""):
        torch.save(self, f"{self.model_dir}/{MODEL_BASE_FILENAME}_{self.transitions_total}{suffix}.pckl")
        print("Saved model.")

    @staticmethod
    def load(env_name, model_name, model_version=None) -> Agent:
        model_dir = get_model_dir(env_name, model_name)

        if not os.path.exists(model_dir):
            raise ValueError(f"No model '{model_name}' found for environment '{env_name}'.")

        if model_version is not None:
            model_save_file = f"{model_dir}/{model_version}.pckl"
            if not os.path.exists(model_save_file):
                raise ValueError(f"No model version '{model_version}' found for model '{model_name}' "
                                 f"and environment '{env_name}'.")

        else:  # pick most recent checkpoint file
            model_save_files = glob.glob(f"{model_dir}/{MODEL_BASE_FILENAME}*.pckl")
            if len(model_save_files) == 0:
                if os.path.exists(f"{model_dir}/result.pckl"):
                    model_save_file = f"{model_dir}/result.pckl"
                else:
                    raise ValueError(f"Model '{model_name}' (of env '{env_name}') has no saved versions.")
            else:
                model_save_file = model_save_files[-1]

        option_critic = torch.load(model_save_file)
        option_critic.running = True
        return option_critic

    def get_current_oc_env_state(self, env: Union[Env, OCAtari]):
        """Returns the object-centric representation of the provided environment's
        current state. It is defined as the normalized vector of x,y,dx,dy properties
        of each non-HUD game object."""
        padded_object_list = pad_object_list(env.objects, self.max_object_counts)
        object_matrix = objects_to_matrix(padded_object_list)
        object_matrix_normalized = normalize_object_matrix(object_matrix)
        return object_matrix_normalized.flatten()


def get_model_dir(env_name, model_name):
    return MODEL_BASE_PATH + f"{env_name}/{model_name}"
