from option_critic import OptionCritic
from utils import make_env

env_name = "ALE/Seaquest-v5"
model_name = "fameskip"

env, is_atari = make_env(env_name, seed=0, render_mode="rgb_array")

# Load model
option_critic = OptionCritic.load(model_name=model_name, env_name=env.spec.name)
option_critic.play(env)
