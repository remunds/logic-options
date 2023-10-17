import os
from pathlib import Path
import shutil

import yaml
from stable_baselines3.common.logger import configure
import torch as th

from utils.common import hyperparams_to_experiment_name, get_torch_device, ask_to_override_model
from utils.param_schedule import maybe_make_schedule
from envs.common import get_env_identifier, init_train_eval_envs, get_focus_file_path
from options.ppo import OptionsPPO
from utils.callbacks import init_callbacks

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
        config_path: str = "",
        description: str = None):
    th.manual_seed(seed)

    game_identifier = get_env_identifier(environment["name"])

    # Set experiment name
    if name is None:
        name = hyperparams_to_experiment_name(environment_kwargs=environment, seed=seed)

    model_path = Path(OUT_BASE_PATH, game_identifier, name)
    log_path = model_path
    ckpt_path = model_path / "checkpoints"

    if name != "debug" and os.path.exists(ckpt_path):
        ask_to_override_model(model_path)

    device = get_torch_device(cuda)

    object_centric = environment.get("object_centric")
    n_envs = cores
    n_eval_envs = cores
    n_eval_episodes = evaluation.get("n_episodes")
    if n_eval_episodes is None:
        n_eval_episodes = 4 * n_eval_envs
    total_timestamps = int(float(training["total_timesteps"]))
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    train_env, eval_env = init_train_eval_envs(n_train_envs=n_envs,
                                               n_eval_envs=n_eval_envs,
                                               seed=seed,
                                               **environment)

    cb_list = init_callbacks(exp_name=name,
                             total_timestamps=total_timestamps,
                             object_centric=object_centric,
                             n_envs=n_envs,
                             eval_env=eval_env,
                             n_eval_episodes=n_eval_episodes,
                             ckpt_path=ckpt_path,
                             eval_frequency=evaluation["frequency"],
                             eval_render=evaluation["render"],
                             eval_deterministic=evaluation["deterministic"],
                             eval_early_stop=evaluation.get("early_stop"))

    hierarchy_shape = model["hierarchy_shape"]
    print(f"Hierarchy shape {hierarchy_shape}")

    policy_kwargs = {"hierarchy_shape": hierarchy_shape,
                     "normalize_images": not object_centric}
    net_arch = model.get("net_arch")
    if net_arch is not None:
        policy_kwargs["net_arch"] = net_arch

    clip_range = maybe_make_schedule(model["ppo"].pop("clip_range"))
    learning_rate = maybe_make_schedule(training.pop("learning_rate"))

    model = OptionsPPO(
        policy_kwargs=policy_kwargs,
        env=train_env,
        learning_rate=learning_rate,
        termination_regularizer=model["termination_regularizer"],
        clip_range=clip_range,
        **model["ppo"],
        device=device,
        verbose=1,
    )

    new_logger = configure(str(log_path), ["tensorboard"])
    model.set_logger(new_logger)

    # Save config file and prune file to model dir for documentation
    shutil.copy(src=config_path, dst=model_path / "config.yaml")
    if name != "debug":
        os.remove(config_path)
    if object_centric:
        prune_file_path = get_focus_file_path(environment.get("prune_concept"), environment["name"])
        shutil.copy(src=prune_file_path, dst=model_path / "prune.yaml")

    print(f"Running experiment '{name}'")
    if description is not None:
        print(description)
    print(f"Started {type(model).__name__} training with {n_envs} actors and {n_eval_envs} evaluators...")
    model.learn(total_timesteps=total_timestamps, callback=cb_list)


if __name__ == "__main__":
    config_files = os.listdir(QUEUE_PATH)
    while len(config_files) > 0:
        config_path = QUEUE_PATH + config_files[0]
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        run(config_path=config_path, **config)
        config_files = os.listdir(QUEUE_PATH)
