from option_critic import OptionCritic
from utils import make_env

env_name = "CartPole-v1"
model_name = "Experiment_1"

env, is_atari = make_env(env_name, seed=0, render_mode="human")
env.render()

# Load model
option_critic = OptionCritic.load(model_name=model_name, env_name=env_name)
option_critic.play(env)
