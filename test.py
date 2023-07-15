from option_critic import OptionCritic
from utils import make_env

# env_name = "ALE/Seaquest-v5"
# model_name = "object-centric"
env_name = "ALE/Kangaroo-v5"
model_name = "debug"
# model_version = "model_15382246_record_17.0"
object_centric = False

env, is_atari = make_env(env_name, seed=0, render_mode="rgb_array",
                         object_centric=object_centric,
                         framestack=1,
                         frameskip=4)
# TODO: import config to correctly specify env params (like framestack)

# Load model
option_critic = OptionCritic.load(model_name=model_name, env_name=env.spec.name)
option_critic.play(env)
