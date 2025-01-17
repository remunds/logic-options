import os
import shutil
from pathlib import Path
from random import randint

import torch as th
import yaml
from stable_baselines3.common.logger import configure

from logic_options.envs.common import init_train_eval_envs, get_focus_file_path
from logic_options.envs.util import get_env_identifier
from logic_options.options.ppo import OptionsPPO
from logic_options.utils.common import hyperparams_to_experiment_name, get_torch_device, update_yaml
from logic_options.utils.console import bold
from logic_options.utils.param_schedule import maybe_make_schedule

OUT_BASE_PATH = "out/"
QUEUE_PATH = "in/queue/"
CHECKPOINT_FREQUENCY = 10_000_000

#set allowed threads to 1
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"


def run(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Mandatory hyperparams
    environment = config["environment"].copy()
    general = config["general"].copy()
    meta_policy = config["meta_policy"].copy()
    evaluation = config["evaluation"].copy()

    # Optional hyperparams
    name = config.get("name")
    description = config.get("description")
    seed = config.get("seed")
    options = config.get("options")
    device = config.get("device")
    cores = config.get("cores")

    if name is None:
        name = hyperparams_to_experiment_name(environment_kwargs=environment, seed=seed)
        config["name"] = name

    print(f"Found configuration, initializing experiment '{bold(name)}'")
    if description is not None and description != '':
        print(f"Description: {description}")

    if seed is None:
        seed = randint(0, 10_000_000)
        config["seed"] = seed
    th.manual_seed(seed)

    if device is None:
        device = "cpu"
        config["device"] = "cpu"

    if cores is None:
        cores = 4
        config["cores"] = 4

    game_identifier = get_env_identifier(environment["name"])

    # Determine all relevant paths
    model_path = Path(OUT_BASE_PATH, game_identifier, name)
    i = 1
    while os.path.exists(model_path) and name != "debug":
        model_path = Path(OUT_BASE_PATH, game_identifier, name + f"_{i}")
        i += 1
    log_path = model_path
    ckpt_path = model_path / "checkpoints"

    # if name != "debug" and os.path.exists(ckpt_path):
    #     ask_to_override_model(model_path)

    device = get_torch_device(device)

    object_centric = environment.get("object_centric")
    use_scobi = object_centric and (environment.get("prune_concept") == 'default'
                                    or environment.get("reward_mode") in ['human', 'mixed'])
    n_envs = cores
    n_eval_envs = cores
    n_eval_episodes = evaluation.pop("n_episodes")
    if n_eval_episodes is None:
        n_eval_episodes = 4 * n_eval_envs
    total_timestamps = int(float(general.pop("total_timesteps")))
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    logic = meta_policy["logic"]
    hierarchy_shape = general.pop("hierarchy_shape")
    uses_options = len(hierarchy_shape) > 0
    print(f"Hierarchy shape {hierarchy_shape}")

    train_env, eval_env = init_train_eval_envs(n_train_envs=n_envs,
                                               n_eval_envs=n_eval_envs,
                                               seed=seed,
                                               logic=logic,
                                               render_eval=evaluation["render"],
                                               accept_predicates=not uses_options,
                                               **environment)
    # create evaluate callback
    from stable_baselines3.common.callbacks import EvalCallback
    eval_callback = EvalCallback(eval_env, best_model_save_path=model_path, log_path=log_path, eval_freq=1000,
                                    deterministic=True, render=False, n_eval_episodes=n_eval_episodes)
    cb_list = [eval_callback]

    policy_kwargs = {}

    if logic:
        policy_kwargs.update({"env_name": game_identifier,
                              "logic_meta_policy": True,
                              "accepts_predicates": uses_options,
                              "device": device})

    net_arch = general.pop("net_arch")
    if net_arch is not None:
        policy_kwargs["net_arch"] = net_arch

    # Init all meta policy schedules
    meta_policy_clip_range = maybe_make_schedule(meta_policy.pop("policy_clip_range"))
    meta_learning_rate = maybe_make_schedule(meta_policy.pop("learning_rate"))

    if options is not None:
        options_activity_coef = options.pop("activity_coef", 0.0)
        options_kwargs = dict(
            options_policy_clip_range=maybe_make_schedule(options.pop("policy_clip_range")),
            options_learning_rate=maybe_make_schedule(options.pop("learning_rate")),
            options_policy_ent_coef=options["policy_ent_coef"],
            options_value_fn_coef=options["value_fn_coef"],
            options_value_fn_clip_range=options["value_fn_clip_range"],
            options_terminator_ent_coef=options["terminator_ent_coef"],
            options_terminator_clip_range=options["terminator_clip_range"],
            options_termination_reg=options["termination_regularizer"],
            options_activity_coef=options_activity_coef,
        )
    else:
        options_kwargs = dict()

    from stable_baselines3 import PPO
    ppo = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=log_path, device=device)

    new_logger = configure(str(log_path), ["tensorboard"])
    ppo.set_logger(new_logger)

    # Transfer learning with existing components (if specified)

    # Save config file and prune file to model dir for documentation
    with open(model_path / "config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)
    # shutil.copy(src=config_path, dst=model_path / "config.yaml")
    if name != "debug":
        os.remove(config_path)
    if use_scobi:
        prune_file_path = get_focus_file_path(environment.get("prune_concept"), environment["name"])
        shutil.copy(src=prune_file_path, dst=model_path / "prune.yaml")

    print(f"Starting experiment '{bold(name)}' with {type(ppo).__name__} training using {n_envs} actors and {n_eval_envs} evaluators...")
    print(cb_list)
    ppo.learn(total_timesteps=total_timestamps, progress_bar=True)


if __name__ == "__main__":
    # Keep processing config files as long as there are some left
    while config_files := os.listdir(QUEUE_PATH):
        config_path = QUEUE_PATH + config_files[0]
        run(config_path)
    print("No more config files left. Shutting down...")
