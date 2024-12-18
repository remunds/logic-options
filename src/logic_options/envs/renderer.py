import numpy as np
import torch as th
import pygame
from datetime import datetime
import gymnasium as gym
from scobi.core import Environment as ScobiEnv
from typing import Union
from nsfr.nsfr import NSFReasoner

from logic_options.options.ppo import load_agent
from logic_options.logic.env_wrapper import LogicEnvWrapper
from logic_options.utils.render import render_options_overlay
from eval_dist_to_joey import get_distance_to_joey

from stable_baselines3.common.vec_env import unwrap_vec_normalize
from stable_baselines3.common.utils import obs_as_tensor
import time

SCREENSHOTS_BASE_PATH = "out/screenshots/"
PREDICATE_PROBS_COL_WIDTH = 300
CELL_BACKGROUND_DEFAULT = np.array([40, 40, 40])
CELL_BACKGROUND_HIGHLIGHT = np.array([40, 150, 255])


class Renderer:
    window: pygame.Surface
    clock: pygame.time.Clock
    env: Union[gym.Env, LogicEnvWrapper]

    def __init__(self, env_name: str,
                 agent_name: str = None,
                 fps: int = None,
                 shadow_mode=False,
                 deterministic=True,
                 wait_for_input=False,
                 render_oc_overlay=True,
                 render_predicate_probs=False):

        self.fps = fps
        self.shadow_mode = shadow_mode
        self.deterministic = deterministic
        self.wait_for_input = wait_for_input
        self.render_predicate_probs = render_predicate_probs

        print(f"Playing '{env_name}' with {'' if deterministic else 'non-'}deterministic policy.")

        self.model = load_agent(agent_name, env_name,
                                render_mode="rgb_array",
                                render_oc_overlay=render_oc_overlay,
                                reward_mode="human")
        self.uses_options = self.model.hierarchy_size > 0
        self.logic = self.model.policy.logic_meta_policy
        if render_predicate_probs:
            raise RuntimeError("Predicate rendering only possible for a logic model. The specified model is neural.")

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
        # self.env._render_action_heatmap()

        self.action_meanings = self.env.get_action_meanings()
        try:
            self.keys2actions = self.env.get_keys_to_action()
        except Exception:
            if shadow_mode or wait_for_input:
                raise RuntimeError(f"Environment {self.env} has no keys-to-actions mapping.")
            print("Info: No key-to-action mapping found for this env. No manual user control possible.")
            self.keys2actions = None
        self.current_keys_down = set()

        self.vec_norm = unwrap_vec_normalize(vec_env)
        # env.vec_norm = vec_norm
        # env.policy = model.policy.meta_policy

        # if self.logic:
        #     self.model.policy.meta_policy.actor.print_program()

        self.predicates = self.model.policy.meta_policy.predicates if self.logic else None
        self.nsfr_reasoner = self.model.policy.meta_policy.actor if self.logic else None

        self._init_pygame()

        self.running = True
        self.paused = False
        self.fast_forward = False
        self.reset = False

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Environment")
        frame = self.vec_env.render().swapaxes(0, 1)
        self.env_render_shape = frame.shape[:2]
        window_shape = list(self.env_render_shape)
        if self.render_predicate_probs:
            window_shape[0] += PREDICATE_PROBS_COL_WIDTH
        self.window = pygame.display.set_mode(window_shape, pygame.SCALED)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Calibri', 24)

    def run(self):
        # Prepare loop
        option_terminations = th.ones(1, self.model.hierarchy_size, dtype=th.bool, device=self.model.device)
        options = th.zeros(1, self.model.hierarchy_size, dtype=th.long, device=self.model.device)
        length = 0

        obs = self.vec_env.reset()

        while self.running:
            # print(obs[0, 0])

            # (options, actions), _, _ = self.model.forward_all(obs, options, option_terminations, self.deterministic)
            (options, actions), _, _ = self.model.forward_all(obs, options, option_terminations, self.deterministic)
            # log_prob all possible options
            # [env][option] .. [1(dist)]
            prob_tensor = th.tensor([0, 1, 2, 3, 4, 5], device=self.model.device)
            if isinstance(obs, np.ndarray):
                obs = obs_as_tensor(obs, self.model.device)
            if len(self.model.policy.options_hierarchy.options) > 0:
                option_tensor = th.tensor([0, 1, 2], device=self.model.device)
            else:
                option_tensor = th.tensor([0, 1, 2, 3, 4, 5], device=self.model.device)
            meta_probs = self.model.policy.meta_policy(obs)[1].log_prob(option_tensor).exp()
            print(meta_probs)
            if len(self.model.policy.options_hierarchy.options) > 0:
                probs = self.model.policy.options_hierarchy[0][0](obs)[1].log_prob(prob_tensor).exp()
                print(probs)
                probs = self.model.policy.options_hierarchy[0][1](obs)[1].log_prob(prob_tensor).exp()
                print(probs)
                probs = self.model.policy.options_hierarchy[0][2](obs)[1].log_prob(prob_tensor).exp()
                print(probs)


            if self.uses_options and hasattr(self.env, "register_current_option"):
                self.env.register_current_option(options[0])

            self._render()

            if self.shadow_mode:
                if self.uses_options and self.logic:
                    # self.nsfr_reasoner.print_probs(self.nsfr_reasoner.V_T)
                    print("\nProposed next option:", self.predicates[options.squeeze()])

                action = actions.squeeze()
                if self.logic and not self.uses_options:
                    action_str = self.predicates[action]
                elif self.action_meanings is not None:
                    action_str = self.action_meanings[action]
                else:
                    action_str = str(action)

                if self.wait_for_input:
                    print("Proposed next action:", action_str)

            self.reset = False

            # self._handle_user_input()
            # human_action = self._get_action()

            if self.shadow_mode:
                if self.wait_for_input:
                    while human_action is None and self.running and not self.reset:
                        self._handle_user_input()
                        human_action = self._get_action()
                        time.sleep(0.1)

                if not self.running:
                    break  # outer game loop

                if not self.reset:
                    # convert action into some matching predicate
                    # if self.logic:
                    #     predicates = self.env.get_predicates_for_action(human_action)
                    #     if len(predicates) > 0:
                    #         human_action = predicates[0]
                    #     else:
                    #         human_action = None  # NOOP
                    actions[0] = human_action

            # Apply action
            if not self.paused:
                new_obs, reward, dones, _ = self.vec_env.step(actions)

                # game_objects = self.vec_env.envs[0].env.oc_env.objects
                # d = get_distance_to_joey(game_objects)
                # print("Distance to Joey:", d)

                if self.shadow_mode and float(reward) != 0:
                    print(f"Reward {reward[0]:.2f}")

                if self.reset:
                    dones[0] = True
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
            print(f"Pressed keys: {pressed_keys}")
            return self.keys2actions[pressed_keys]
        else:
            return None  # NOOP

    def _handle_user_input(self):

        analyze_termination_keys = {
            pygame.K_v: 0, pygame.K_b: 1, pygame.K_n: 2, pygame.K_m: 3}
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:  # window close button clicked
                self.running = False

            elif event.type == pygame.KEYDOWN:  # keyboard key pressed
                if event.key == pygame.K_p:  # 'P': pause/resume
                    self.paused = not self.paused

                elif event.key == pygame.K_r:  # 'R': reset
                    self.reset = True

                elif event.key == pygame.K_f:  # 'F': fast forward
                    self.fast_forward = True

                elif event.key == pygame.K_t:  # 'T': trigger takeover
                    self.shadow_mode = not self.shadow_mode

                elif event.key == pygame.K_c:  # 'C': capture screenshot
                    file_name = f"{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')}.png"
                    pygame.image.save(self.window, SCREENSHOTS_BASE_PATH + file_name)

                elif event.key in analyze_termination_keys.keys():  # analyze option termination
                    option_pos = analyze_termination_keys[event.key] 
                    if option_pos < len(self.model.policy.options_hierarchy[0]):
                        if self.env.action is not None:
                            # currently rendering action heatmap, turn off
                            self.env.render_action_heatmap_for_policy(action=None)

                        option_to_render = self.model.policy.options_hierarchy[0][option_pos]
                        if self.env.option != option_to_render and self.env.option_pos != option_pos:
                            if self.model.policy_terminator:
                                self.env.render_termination_heatmap_by_policy(self.model.policy, option_pos, self.vec_norm)
                            else:
                                self.env.render_termination_heatmap_of_option(option_to_render, self.vec_norm)
                        else:
                            # if self.model.policy.policy_terminator: 
                            if self.model.policy_terminator: 
                                self.env.render_termination_heatmap_by_policy(None, None)
                            else:
                                self.env.render_termination_heatmap_of_option(None)
                        self._render()
                    else:
                        print(f"No top-level option at pos {option_pos}.")

                elif pygame.K_0 <= event.key <= pygame.K_9:  # analyze option invocation probability
                    print(f"Option invocation probability for option {event.key - pygame.K_0}:")
                    option_pos = event.key - pygame.K_0
                    if option_pos < len(self.model.policy.options_hierarchy[0]):
                        if self.env.option is not None:
                            # currently rendering option termination heatmap, turn off
                            self.env.render_termination_heatmap_of_option(None)
                        if self.env.option_pos is not None:
                            # currently rendering option termination heatmap, turn off
                            self.env.render_termination_heatmap_by_policy(None, None)

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

                elif event.key == pygame.K_f:  # 'F': fast forward
                    self.fast_forward = False

    def _render(self):
        self.window.fill((20, 20, 20))  # clear the entire window
        self._render_env()
        if self.render_predicate_probs:
            self._render_predicate_probs()

        # TODO:
        # render_options_overlay(image, option_trace=options[0].tolist())

        pygame.display.flip()
        pygame.event.pump()
        if not self.fast_forward:
            self.clock.tick(self.fps)

    def _render_env(self):
        frame = self.vec_env.render().swapaxes(0, 1)
        frame_surface = pygame.Surface(self.env_render_shape)
        pygame.pixelcopy.array_to_surface(frame_surface, frame)
        self.window.blit(frame_surface, (0, 0))

    def _render_predicate_probs(self):
        anchor = (self.env_render_shape[0] + 10, 25)

        nsfr: NSFReasoner = self.model.policy.meta_policy.actor
        pred_vals = {pred: nsfr.get_predicate_valuation(pred, initial_valuation=False) for pred in nsfr.prednames}
        for i, (pred, val) in enumerate(pred_vals.items()):
            # Render cell background
            color = val * CELL_BACKGROUND_HIGHLIGHT + (1 - val) * CELL_BACKGROUND_DEFAULT
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * 35,
                PREDICATE_PROBS_COL_WIDTH - 12,
                28
            ])

            text = self.font.render(str(f"{val:.3f} - {pred}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, 25 + i * 35)
            self.window.blit(text, text_rect)
