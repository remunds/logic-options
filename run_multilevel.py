import os
from pathlib import Path
import shutil

import yaml
from stable_baselines3.common.logger import configure
import torch as th

from utils import maybe_make_schedule, get_experiment_name_from_hyperparams, init_envs, get_torch_device, \
    get_atari_identifier, get_pruned_focus_file_from_env_name, FOCUS_FILES_DIR
from options_ppo import OptionsPPO
from callbacks import init_callbacks

OUT_BASE_PATH = "out/"
QUEUE_PATH = "in/queue/"


def run(name: str = None,
        cuda: bool = True,
        cores: int = 1,
        seed: int = 0,
        environment: dict = None,
        model: dict = None,
        training: dict = None,
        evaluation: dict = None,
        config_path: str = ""):
    th.manual_seed(seed)

    game_identifier = get_atari_identifier(environment["name"])

    # Set experiment name
    if name is None:
        name = get_experiment_name_from_hyperparams(environment_kwargs=environment, seed=seed)

    model_path = Path(OUT_BASE_PATH, game_identifier, name)

    device = get_torch_device(cuda)

    object_centric = environment["object_centric"]
    n_envs = cores
    n_eval_envs = cores
    total_timestamps = int(float(training["total_timesteps"]))
    log_path = model_path / "logs"
    ckpt_path = model_path / "checkpoints"
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    train_env, eval_env = init_envs(n_envs=n_envs, n_eval_envs=n_eval_envs, seed=seed, **environment)

    cb_list = init_callbacks(exp_name=name,
                             total_timestamps=total_timestamps,
                             object_centric=object_centric,
                             n_envs=n_envs,
                             eval_env=eval_env,
                             n_eval_episodes=4 * n_eval_envs,
                             ckpt_path=ckpt_path,
                             eval_frequency=evaluation["frequency"],
                             eval_render=evaluation["render"],
                             eval_deterministic=evaluation["deterministic"])

    policy_kwargs = {"options_hierarchy": model["options_hierarchy"],
                     "net_arch": model["net_arch"]}
    clip_range = maybe_make_schedule(model["ppo"].pop("clip_range"))
    learning_rate = maybe_make_schedule(training.pop("learning_rate"))

    model = OptionsPPO(
        policy_kwargs=policy_kwargs,
        env=train_env,
        learning_rate=learning_rate,
        clip_range=clip_range,
        **model["ppo"],
        device=device,
        verbose=1,
    )

    new_logger = configure(str(log_path), ["tensorboard"])
    model.set_logger(new_logger)

    # Save config file and prune file to model dir for documentation
    shutil.copy(src=config_path, dst=model_path / "config.yaml")
    prune_file_path = Path(FOCUS_FILES_DIR, get_pruned_focus_file_from_env_name(environment["name"]))
    shutil.copy(src=prune_file_path, dst=model_path / "prune.yaml")

    print(f"Experiment name: {name}")
    print(f"Started {type(model).__name__} training with {n_envs} actors and {n_eval_envs} evaluators...")
    model.learn(total_timesteps=total_timestamps, callback=cb_list)


if __name__ == "__main__":
    config_files = os.listdir(QUEUE_PATH)
    while len(config_files) > 0:
        config_path = QUEUE_PATH + config_files[0]
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        run(config_path=config_path, **config)
        os.remove(config_path)
        config_files = os.listdir(QUEUE_PATH)
