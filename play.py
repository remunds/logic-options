import numpy as np
import torch as th

from options.ppo import load_agent
from utils.render import render_options_overlay


if __name__ == "__main__":
    name = "4-floor/reward-v1-1"
    env_name = "MeetingRoom"

    deterministic = False
    print(f"Playing with {'' if deterministic else 'non-'}deterministic policy.")

    model = load_agent(name, env_name, render_mode="rgb_array", render_oc_overlay=True, reward_mode="human")
    env = model.get_env()

    # Prepare loop
    obs = env.reset()
    option_terminations = th.ones(1, model.hierarchy_size).type(th.BoolTensor)
    options = th.zeros(1, model.hierarchy_size).type(th.LongTensor)
    length = 0

    while True:
        (options, actions), _, _ = model.forward_all(obs, options, option_terminations, deterministic)

        new_obs, reward, dones, _ = env.step(actions)

        # if reward[0] != 0:
        #     print("Reward:", reward[0])

        image = env.render()
        render_options_overlay(image,
                               option_trace=options[0].tolist(),
                               fps=env.metadata.get("video.frames_per_second"))

        option_terminations, _ = model.forward_all_terminators(new_obs, options)
        option_terminations[dones] = True

        obs = new_obs
        length += 1

        if np.any(dones):
            rewards = env.envs[0].get_episode_rewards()
            if len(rewards) > 0:
                ret = rewards[-1]
                print(f"Return: {ret} - Length {length}")
                length = 0
