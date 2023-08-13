import numpy as np
import torch as th

from options_ppo import load_agent

name = "debug"
env_name = "ALE/Pong-v5"

deterministic = True

model = load_agent(name, env_name, render_mode="human")
env = model.get_env()

# Prepare loop
obs = env.reset()
option_terminations = th.ones(1, model.hierarchy_size).type(th.BoolTensor)
options = th.zeros(1, model.hierarchy_size).type(th.LongTensor)

while True:
    (options, actions), _, _ = model.forward_all(obs, options, option_terminations, deterministic)

    new_obs, _, dones, _ = env.step(actions)

    option_terminations, _ = model.forward_all_terminators(new_obs, options)
    option_terminations[dones] = True

    obs = new_obs

    if np.any(dones):
        ret = env.envs[0].get_episode_rewards()[-1]
        print(f"Return: {ret}")

    env.render()
