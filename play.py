import numpy as np
import torch as th
import pygame
from datetime import datetime

from options.ppo import load_agent
from utils.render import render_options_overlay

from stable_baselines3.common.vec_env import unwrap_vec_normalize

SCREENSHOTS_BASE_PATH = "out/screenshots/"


if __name__ == "__main__":
    name = "4-floor/options/logic-scratch-less-entropy"
    # name = "4-floor/options/logic-pre-trained-less-entropy"
    # name = "4-floor/options/neural-hierarchy-more-tn-entropy"
    # name = "4-floor/options/logic-pre-trained-even-smaller-lr"
    env_name = "MeetingRoom"
    shadow_mode = True
    deterministic = True

    print(f"Playing with {'' if deterministic else 'non-'}deterministic policy.")

    model = load_agent(name, env_name,
                       render_mode="rgb_array",
                       render_oc_overlay=True,
                       reward_mode="human",
                       device="cuda")
    uses_options = model.hierarchy_size > 0
    logic = model.policy.logic_meta_policy
    vec_env = model.get_env()
    vec_env.metadata["video.frames_per_second"] = 4
    env = vec_env.envs[0].env

    env.set_render_option_history(True)
    # env.render_termination_heatmap(True)
    # env.render_action_heatmap(True)

    keys2actions = env.get_keys_to_action()
    action_meanings = env.get_action_meanings()

    vec_norm = unwrap_vec_normalize(vec_env)
    # env.vec_norm = vec_norm
    # env.policy = model.policy.meta_policy

    if logic:
        model.policy.meta_policy.actor.print_program()

    predicates = model.policy.meta_policy.predicates if logic else None
    nsfr_reasoner = model.policy.meta_policy.actor if logic else None

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

        if uses_options:
            env.register_current_option(options[0])

        image = vec_env.render()
        render_options_overlay(image,
                               option_trace=options[0].tolist(),
                               fps=vec_env.metadata.get("video.frames_per_second"))

        if shadow_mode:
            if uses_options and logic:
                nsfr_reasoner.print_probs(nsfr_reasoner.V_T)
                print("\nProposed next option:", predicates[options.squeeze()])

            action = actions.squeeze()
            if logic and not uses_options:
                action = predicates[action]
            print("Proposed next action:", action_meanings[action])

        reset = False
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:  # keyboard key pressed
                if event.key == pygame.K_r:  # 'R': reset
                    reset = True

        if shadow_mode:
            human_action = None
            while human_action is None and running and not reset:
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:  # window close button clicked
                        running = False

                    elif event.type == pygame.KEYDOWN:  # keyboard key pressed
                        if event.key == pygame.K_r:  # 'R': reset
                            reset = True
                        elif event.key == pygame.K_c:  # 'C': capture screenshot
                            file_name = f"{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')}.png"
                            pygame.image.save(env.window, SCREENSHOTS_BASE_PATH + file_name)
                        elif pygame.K_0 <= event.key <= pygame.K_9:  # analyze option termination
                            option_pos = event.key - pygame.K_0
                            if option_pos < len(model.policy.options_hierarchy[0]):
                                option_to_render = model.policy.options_hierarchy[0][option_pos]
                                if env.option != option_to_render:
                                    env.render_termination_heatmap_of_option(option_to_render, vec_norm)
                                else:
                                    env.render_termination_heatmap_of_option(None)
                                image = vec_env.render()
                                render_options_overlay(image,
                                                       option_trace=options[0].tolist(),
                                                       fps=vec_env.metadata.get("video.frames_per_second"))
                            else:
                                print(f"No top-level option at pos {option_pos}.")
                        elif pygame.K_KP1 <= event.key <= pygame.K_KP0:  # analyze option invocation probability
                            option_pos = (event.key - pygame.K_KP1 + 1) % 10
                            if option_pos < len(model.policy.options_hierarchy[0]):
                                if env.action != option_pos:
                                    env.render_action_heatmap_for_policy(policy=model.policy.meta_policy,
                                                                         action=option_pos,
                                                                         vec_norm=vec_norm)
                                else:
                                    env.render_action_heatmap_for_policy(action=None)
                                image = vec_env.render()
                                render_options_overlay(image,
                                                       option_trace=options[0].tolist(),
                                                       fps=vec_env.metadata.get("video.frames_per_second"))
                            else:
                                print(f"No top-level option at pos {option_pos}.")
                        else:
                            human_action = keys2actions.get((event.key,))
            if not running:
                break
            if not reset:
                actions[0] = human_action

        new_obs, reward, dones, _ = vec_env.step(actions)

        if shadow_mode and float(reward) != 0:
            print(f"Reward {reward[0]:.2f}")

        if reset:
            dones[:] = True
            new_obs = vec_env.reset()

        option_terminations, _ = model.forward_all_terminators(new_obs, options)
        option_terminations[dones] = True

        obs = new_obs
        length += 1

        # image = vec_env.render()
        # render_options_overlay(image,
        #                        option_trace=options[0].tolist(),
        #                        fps=vec_env.metadata.get("video.frames_per_second"))

        if np.any(dones):
            rewards = vec_env.envs[0].get_episode_rewards()
            if len(rewards) > 0:
                ret = rewards[-1]
                print(f"Return: {ret} - Length {length}")
                length = 0

        # if terminated or truncated:
        #     env.reset()
