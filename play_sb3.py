import numpy as np
import yaml

from options_ppo import MODELS_BASE_PATH
from utils import get_atari_identifier
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.ppo.ppo import PPO

name = "run-0"
env_name = "ALE/Tennis-v5"

deterministic = False

env_identifier = get_atari_identifier(env_name)
model_dir = f"{MODELS_BASE_PATH}{env_identifier}/{name}/"
config_path = model_dir + "config.yaml"
model_path = model_dir + "checkpoints/best_model.zip"

with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)

device = "cuda" if config["cuda"] else "cpu"

env = make_atari_env(env_name, n_envs=1, seed=config["seed"],
                     wrapper_kwargs={"frame_skip": config["environment"].get("frameskip"),
                                     "terminal_on_life_loss": False},
                     env_kwargs={"render_mode": "human"})
env = VecFrameStack(env, n_stack=config["environment"].get("framestack"))

model = PPO.load(model_path, env=env, verbose=1, device=device)

obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=deterministic)
    # action = [env.action_space.sample()]
    obs, _, done, _ = env.step(action)
    if np.any(done):
        ret = env.envs[0].get_episode_rewards()[-1]
        print(f"Return: {ret}")
