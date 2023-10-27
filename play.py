import numpy as np
import torch as th
import pygame
from datetime import datetime

from options.ppo import load_agent
from utils.render import render_options_overlay

from stable_baselines3.common.vec_env import unwrap_vec_normalize

SCREENSHOTS_BASE_PATH = "out/screenshots/"


if __name__ == "__main__":
    name = "4-floor/options/less-entropy-1"
    env_name = "MeetingRoom"
    shadow_mode = True
    deterministic = False

    print(f"Playing with {'' if deterministic else 'non-'}deterministic policy.")

    model = load_agent(name, env_name, render_mode="rgb_array",
                       render_oc_overlay=True,
                       reward_mode="human")
    logic = model.policy.logic_meta_policy
    vec_env = model.get_env()
    env = vec_env.envs[0].env

    keys2actions = env.get_keys_to_action()
    action_meanings = env.get_action_meanings()

    vec_norm = unwrap_vec_normalize(vec_env)
    env.vec_norm = vec_norm

    if logic:
        model.policy.meta_policy.actor.print_program()

    # nsfr_reasoner = model.meta_policy.actor if logic else None
    # predicates = model.meta_policy.predicates if logic else None

    # Prepare loop
    obs = vec_env.reset()
    vec_env.render()
    option_terminations = th.ones(1, model.hierarchy_size, dtype=th.bool, device=model.device)
    options = th.zeros(1, model.hierarchy_size, dtype=th.long, device=model.device)
    length = 0

    image = vec_env.render()
    render_options_overlay(image,
                           option_trace=options[0].tolist(),
                           fps=vec_env.metadata.get("video.frames_per_second"))

    running = True
    while running:
        (options, actions), _, _ = model.forward_all(obs, options, option_terminations, deterministic)
        print("Proposed next action:", action_meanings[actions.squeeze()])
        # nsfr_reasoner.print_probs(nsfr_reasoner.V_T)

        if shadow_mode:
            human_action = None
            while human_action is None and running:
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:  # window close button clicked
                        running = False

                    elif event.type == pygame.KEYDOWN:  # keyboard key pressed
                        if event.key == pygame.K_r:  # 'R': reset
                            vec_env.reset()
                        elif event.key == pygame.K_c:  # 'C': capture screenshot
                            file_name = f"{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')}.png"
                            pygame.image.save(vec_env.window, SCREENSHOTS_BASE_PATH + file_name)
                        elif event.key == pygame.K_0:  # '0': analyze option 0
                            env.option = model.policy.options_hierarchy[0][0]
                        elif event.key == pygame.K_1:  # '1': analyze option 1
                            env.option = model.policy.options_hierarchy[0][1]
                        elif event.key == pygame.K_2:  # '2': analyze option 2
                            env.option = model.policy.options_hierarchy[0][2]
                        elif event.key == pygame.K_3:  # '3': analyze option 3
                            env.option = model.policy.options_hierarchy[0][3]
                        else:
                            human_action = keys2actions.get((event.key,))
            if not running:
                break
            actions[0] = human_action

        new_obs, reward, dones, _ = vec_env.step(actions)

        if float(reward) != 0:
            print(f"Reward {reward}")

        option_terminations, _ = model.forward_all_terminators(new_obs, options)
        option_terminations[dones] = True

        obs = new_obs
        length += 1

        image = vec_env.render()
        render_options_overlay(image,
                               option_trace=options[0].tolist(),
                               fps=vec_env.metadata.get("video.frames_per_second"))

        if np.any(dones):
            rewards = vec_env.envs[0].get_episode_rewards()
            if len(rewards) > 0:
                ret = rewards[-1]
                print(f"Return: {ret} - Length {length}")
                length = 0

        # if terminated or truncated:
        #     env.reset()
