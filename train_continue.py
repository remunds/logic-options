from pathlib import Path

import yaml
import torch as th

from logic_options.envs.common import init_train_eval_envs
from logic_options.envs.util import get_atari_identifier
from logic_options.utils.callbacks import init_callbacks
from logic_options.options.ppo import load_agent

ENV_NAME = "ALE/Kangaroo-v5"
MODEL_NAME = "reward-shaping/v2-2"

OUT_BASE_PATH = "out/"


def run():
    game_identifier = get_atari_identifier(ENV_NAME)

    model_dir = Path(OUT_BASE_PATH, game_identifier, MODEL_NAME)

    # Retrieve experiment configuration
    config_path = model_dir / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    n_envs = config["cores"]

    model = load_agent(model_dir=model_dir, best_model=False, n_envs=n_envs, train=True)

    environment = config["environment"]
    evaluation = config["evaluation"]
    training = config["training"]

    th.manual_seed(config["seed"])

    object_centric = environment["object_centric"]
    n_eval_envs = config["cores"]
    total_timestamps = int(float(training["total_timesteps"]))
    ckpt_path = model_dir / "checkpoints"

    _, eval_env = init_train_eval_envs(n_train_envs=0,
                                       n_eval_envs=n_eval_envs,
                                       seed=config["seed"],
                                       **environment)

    cb_list = init_callbacks(exp_name=MODEL_NAME,
                             total_timestamps=total_timestamps,
                             may_use_reward_shaping=object_centric,
                             n_envs=n_envs,
                             eval_env=eval_env,
                             n_eval_episodes=4 * n_eval_envs,
                             ckpt_path=ckpt_path,
                             eval_kwargs=evaluation)

    remaining_timesteps = total_timestamps - model.num_timesteps

    if remaining_timesteps <= 0:
        print("No timesteps remain for training, it was already finished.")
        return

    print(f"Continuing experiment {MODEL_NAME}.")
    print(f"Started {type(model).__name__} training for {remaining_timesteps} steps "
          f"with {n_envs} actors and {n_eval_envs} evaluators...")
    model.learn(total_timesteps=remaining_timesteps,
                callback=cb_list,
                log_interval=None,
                reset_num_timesteps=False)


if __name__ == "__main__":
    run()
