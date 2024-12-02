from datetime import datetime
import os

import pygame
import torch as th
from gymnasium import spaces

from logic_options.envs.common import make_logic_env
from logic_options.options.agent import OptionsAgent

SCREENSHOTS_BASE_PATH = "../../out/screenshots/"


if __name__ == "__main__":
    # Set working dir to project root
    os.chdir("../..")

    env = make_logic_env("MeetingRoom",
                         render_mode="human",
                         walls_fixed=False,
                         accept_predicates=False,
                         goal="target")()
    env.metadata['video.frames_per_second'] = 60
    env.reset()
    keys2actions = env.get_keys_to_action()
    action_meanings = env.get_action_meanings()

    shadowed_agent = OptionsAgent(observation_space=env.observation_space,
                                  action_space=None,
                                  lr_schedule=lambda x: 0,
                                  hierarchy_shape=[],
                                  logic_meta_policy=True,
                                  env_name=env.game_name)
    nsfr_reasoner = shadowed_agent.meta_policy.actor
    predicates = shadowed_agent.meta_policy.predicates

    option_terminations = th.ones(1, shadowed_agent.hierarchy_size).type(th.BoolTensor)
    options = th.zeros(1, shadowed_agent.hierarchy_size).type(th.LongTensor)

    running = True
    while running:
        action = None
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:  # window close button clicked
                running = False

            elif event.type == pygame.KEYDOWN:  # keyboard key pressed
                if event.key == pygame.K_r:  # 'R': reset
                    env.reset()
                elif event.key == pygame.K_c:  # 'C': capture screenshot
                    file_name = f"{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')}.png"
                    pygame.image.save(env.window, SCREENSHOTS_BASE_PATH + file_name)
                else:
                    action = keys2actions.get((event.key,))

        if action is not None:
            obs, reward, _, terminated, truncated = env.step(action)

            if float(reward) != 0:
                print(f"Reward {reward}")

            if shadowed_agent is not None:
                (options, actions), values, log_probs = \
                    shadowed_agent.forward_all(obs.unsqueeze(dim=0), options, option_terminations)
                print("Proposed next action:", predicates[actions.squeeze()])
                # nsfr_reasoner.print_probs(nsfr_reasoner.V_T)

            if terminated or truncated:
                env.reset()

        env.render()
