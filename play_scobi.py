import numpy as np
from stable_baselines3 import PPO
from utils import get_pruned_focus_file_from_env_name, make_scobi_env

MODELS_BASE_PATH = "out/scobi_sb3"

model_name = "rl-zoo-oc-mixed"
device = "cpu"

# Environment params
name = "ALE/Pong-v5"
object_centric = True
reward_mode = "mixed"
prune_concept = "default"
exclude_properties = True
framestack = 4
frameskip = 1

model_path = f"{MODELS_BASE_PATH}/{model_name}/checkpoints/best_model.zip"


pruned_ff_name = get_pruned_focus_file_from_env_name(name)
env = make_scobi_env(name=name, reward_mode=reward_mode,
                     focus_dir="in/focusfiles", pruned_ff_name=pruned_ff_name,
                     exclude_properties=exclude_properties)()

model = PPO.load(model_path, env=env, verbose=1, render_mode="human", device=device)

env = model.get_env()
obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = env.step(action)
    if np.any(dones):
        ret = env.envs[0].get_episode_rewards()[-1]
        print(f"Return: {ret}")
    env.render("human")
