import numpy as np
import torch as th
import pygame
from datetime import datetime
import gymnasium as gym
from scobi.core import Environment as ScobiEnv

from options.ppo import load_agent
from utils.render import render_options_overlay

from stable_baselines3.common.vec_env import unwrap_vec_normalize

SCREENSHOTS_BASE_PATH = "out/screenshots/"


class Renderer:
    window: pygame.Surface
    clock: pygame.time.Clock
    env: gym.Env

    def __init__(self, env_name: str,
                 agent_name: str = None,
                 fps: int = None,
                 shadow_mode=False,
                 deterministic=True,
                 wait_for_input=False):

        self.fps = fps
        self.shadow_mode = shadow_mode
        self.deterministic = deterministic
        self.wait_for_input = wait_for_input

        print(f"Playing '{env_name}' with {'' if deterministic else 'non-'}deterministic policy.")

        self.model = load_agent(agent_name, env_name,
                                render_mode="rgb_array",
                                render_oc_overlay=True,
                                reward_mode="human",
                                device="cuda")
        self.uses_options = self.model.hierarchy_size > 0
        self.logic = self.model.policy.logic_meta_policy
        vec_env = self.model.get_env()

        if fps is None:
            fps = vec_env.metadata.get("video.frames_per_second")
        if fps is None:
            fps = 15
        self.fps = fps

        self.env = vec_env.envs[0].env
        if isinstance(self.env, ScobiEnv):
            self.env = self.env.oc_env.env.unwrapped

        self.vec_env = vec_env
        self.vec_env.reset()

        if self.uses_options and hasattr(self.env, "set_render_option_history"):
            self.env.set_render_option_history(True)
        # env.render_termination_heatmap(True)
        # env.render_action_heatmap(True)

        self.action_meanings = self.env.get_action_meanings()
        if hasattr(self.env, "get_keys_to_action"):
            self.keys2actions = self.env.get_keys_to_action()
        else:
            self.keys2actions = None
        self.current_keys_down = set()

        self.vec_norm = unwrap_vec_normalize(vec_env)
        # env.vec_norm = vec_norm
        # env.policy = model.policy.meta_policy

        if self.logic:
            self.model.policy.meta_policy.actor.print_program()

        self.predicates = self.model.policy.meta_policy.predicates if self.logic else None
        self.nsfr_reasoner = self.model.policy.meta_policy.actor if self.logic else None

        self._init_pygame()

        self.running = True
        self.paused = False
        self.reset = False

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Environment")
        frame = self.env.render()
        frame = frame.swapaxes(0, 1)
        self.window = pygame.display.set_mode(frame.shape[:2], pygame.SCALED)
        self.clock = pygame.time.Clock()

    def run(self):
        # Prepare loop
        option_terminations = th.ones(1, self.model.hierarchy_size, dtype=th.bool, device=self.model.device)
        options = th.zeros(1, self.model.hierarchy_size, dtype=th.long, device=self.model.device)
        length = 0

        obs = self.vec_env.reset()

        while self.running:
            (options, actions), _, _ = self.model.forward_all(obs, options, option_terminations, self.deterministic)

            if self.uses_options and hasattr(self.env, "register_current_option"):
                self.env.register_current_option(options[0])

            self._render()

            if self.shadow_mode:
                if self.uses_options and self.logic:
                    self.nsfr_reasoner.print_probs(self.nsfr_reasoner.V_T)
                    print("\nProposed next option:", self.predicates[options.squeeze()])

                action = actions.squeeze()
                if self.logic and not self.uses_options:
                    action = self.predicates[action]
                if self.action_meanings is not None:
                    print("Proposed next action:", self.action_meanings[action])

            self.reset = False

            self._handle_user_input()
            human_action = self._get_action()

            if self.shadow_mode:
                if self.wait_for_input:
                    while human_action == 0 and self.running and not self.reset:
                        self._handle_user_input()
                        human_action = self._get_action()

                if not self.running:
                    break  # outer game loop

                if not self.reset:
                    actions[0] = human_action

            # Apply action
            if not self.paused:
                new_obs, reward, dones, _ = self.vec_env.step(actions)

                if self.shadow_mode and float(reward) != 0:
                    print(f"Reward {reward[0]:.2f}")

                if self.reset:
                    dones[:] = True
                    new_obs = self.vec_env.reset()

                option_terminations, _ = self.model.forward_all_terminators(new_obs, options)
                option_terminations[dones] = True

                obs = new_obs
                length += 1

                if np.any(dones):
                    rewards = self.vec_env.envs[0].get_episode_rewards()
                    if len(rewards) > 0:
                        ret = rewards[-1]
                        print(f"Return: {ret} - Length {length}")
                        length = 0

            # if terminated or truncated:
            #     env.reset()

        pygame.quit()

    def _get_action(self):
        pressed_keys = list(self.current_keys_down)
        pressed_keys.sort()
        pressed_keys = tuple(pressed_keys)
        if pressed_keys in self.keys2actions.keys():
            return self.keys2actions[pressed_keys]
        else:
            return 0  # NOOP

    def _handle_user_input(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:  # window close button clicked
                self.running = False

            elif event.type == pygame.KEYDOWN:  # keyboard key pressed
                if event.key == pygame.K_p:  # 'P': pause/resume
                    self.paused = not self.paused

                elif event.key == pygame.K_r:  # 'R': reset
                    self.reset = True

                elif event.key == pygame.K_c:  # 'C': capture screenshot
                    file_name = f"{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')}.png"
                    pygame.image.save(self.window, SCREENSHOTS_BASE_PATH + file_name)

                elif pygame.K_0 <= event.key <= pygame.K_9:  # analyze option termination
                    option_pos = event.key - pygame.K_0
                    if option_pos < len(self.model.policy.options_hierarchy[0]):
                        option_to_render = self.model.policy.options_hierarchy[0][option_pos]
                        if self.env.option != option_to_render:
                            self.env.render_termination_heatmap_of_option(option_to_render, self.vec_norm)
                        else:
                            self.env.render_termination_heatmap_of_option(None)
                        self._render()
                    else:
                        print(f"No top-level option at pos {option_pos}.")

                elif pygame.K_KP1 <= event.key <= pygame.K_KP0:  # analyze option invocation probability
                    option_pos = (event.key - pygame.K_KP1 + 1) % 10
                    if option_pos < len(self.model.policy.options_hierarchy[0]):
                        if self.env.action != option_pos:
                            self.env.render_action_heatmap_for_policy(policy=self.model.policy.meta_policy,
                                                                      action=option_pos,
                                                                      vec_norm=self.vec_norm)
                        else:
                            self.env.render_action_heatmap_for_policy(action=None)
                        self._render()
                    else:
                        print(f"No top-level option at pos {option_pos}.")

                elif (event.key,) in self.keys2actions.keys():  # env action
                    self.current_keys_down.add(event.key)

            elif event.type == pygame.KEYUP:  # keyboard key released
                if (event.key,) in self.keys2actions.keys():
                    self.current_keys_down.remove(event.key)

    def _render(self):
        self.window.fill((0, 0, 0))  # clear the entire window
        self._render_env()

        # TODO:
        # render_options_overlay(image, option_trace=options[0].tolist())

        pygame.display.flip()
        pygame.event.pump()
        self.clock.tick(self.fps)

    def _render_env(self):
        frame = self.env.render()
        frame = frame.swapaxes(0, 1)
        pygame.pixelcopy.array_to_surface(self.window, frame)
