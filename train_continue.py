from pathlib import Path
import os
import yaml
import torch as th

from logic_options.envs.common import init_train_eval_envs
from logic_options.utils.common import get_torch_device
from logic_options.envs.util import get_atari_identifier
from logic_options.utils.callbacks import init_callbacks
from logic_options.options.ppo import load_agent
from logic_options.utils.console import bold

from stable_baselines3.common.logger import configure

from random import randint

from logic_options.utils.param_schedule import maybe_make_schedule

ENV_NAME = "ALE/Seaquest-v5"
MODEL_NAME = "pretrain_enemies_best"

OUT_BASE_PATH = "out/"
CHECKPOINT_FREQUENCY = 1_000_000

#set allowed threads to 4
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

th.set_num_threads(4)


def run():
    #NOTE: currently, logic policies can only be resumed on cpu

    game_identifier = get_atari_identifier(ENV_NAME)

    model_dir = Path(OUT_BASE_PATH, game_identifier, MODEL_NAME)

    # Retrieve experiment configuration
    config_path = model_dir / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    # Mandatory hyperparams
    environment = config["environment"].copy()
    general = config["general"].copy()
    meta_policy = config["meta_policy"].copy()
    evaluation = config["evaluation"].copy()
    if config.get("options") is not None:
        options = config["options"].copy()

    # Optional hyperparams
    name = config.get("name")
    description = config.get("description")
    seed = config.get("seed")
    device = config.get("device")
    cores = config.get("cores")

    print(f"Found configuration, loading experiment '{bold(name)}'")
    if description is not None and description != '':
        print(f"Description: {description}")

    if seed is None:
        seed = randint(0, 10_000_000)
        config["seed"] = seed
    th.manual_seed(seed)

    if device is None:
        device = "cpu"
        config["device"] = "cpu"
        device = get_torch_device(device)

    if cores is None:
        cores = 4
        config["cores"] = 4

    n_envs = config["cores"]


    model = load_agent(model_dir=model_dir, best_model=False, n_envs=n_envs, train=True, device=device)

    object_centric = environment.get("object_centric")
    reward_shaping = object_centric and (environment.get("prune_concept") == 'default'
                                    or environment.get("reward_mode") in ['human', 'mixed'])
    n_envs = cores
    n_eval_envs = cores
    n_eval_episodes = evaluation.pop("n_episodes")
    if n_eval_episodes is None:
        n_eval_episodes = 4 * n_eval_envs
    total_timestamps = int(float(general.pop("total_timesteps")))

    logic = meta_policy["logic"]
    hierarchy_shape = general.pop("hierarchy_shape")
    uses_options = len(hierarchy_shape) > 0
    print(f"Hierarchy shape {hierarchy_shape}")

    log_path = model_dir
    ckpt_path = model_dir / "checkpoints"

    _, eval_env = init_train_eval_envs(n_train_envs=n_envs,
                                               n_eval_envs=n_eval_envs,
                                               seed=seed,
                                               logic=logic,
                                               render_eval=evaluation["render"],
                                               accept_predicates=not uses_options,
                                               **environment)

    cb_list = init_callbacks(exp_name=name,
                             total_timestamps=total_timestamps,
                             may_use_reward_shaping=reward_shaping,
                             n_envs=n_envs,
                             eval_env=eval_env,
                             n_eval_episodes=n_eval_episodes,
                             ckpt_path=ckpt_path,
                             eval_kwargs=evaluation,
                             checkpoint_frequency=CHECKPOINT_FREQUENCY)

    remaining_timesteps = total_timestamps - model.num_timesteps

    if remaining_timesteps <= 0:
        print("No timesteps remain for training, it was already finished.")
        return

    model.tensorboard_log = str(log_path)
    new_logger = configure(str(log_path), ["tensorboard"])
    # model.set_env(train_env)
    model.set_logger(new_logger)

    meta_policy_clip_range = maybe_make_schedule(meta_policy.pop("policy_clip_range"))
    prev_meta_learning_rate =  model.meta_learning_rate(0.005) # call schedule with 0.0 progress remaining
    new_meta_lr_args = meta_policy.pop("learning_rate")
    new_meta_lr_args["initial_value"] = prev_meta_learning_rate
    meta_learning_rate = maybe_make_schedule(new_meta_lr_args)

    if uses_options: 
        options_clip_range = maybe_make_schedule(options.pop("policy_clip_range"))
        prev_options_learning_rate = model.options_learning_rate(0.005) # call schedule with 0.0 progress remaining
        new_options_lr_args = options.pop("learning_rate")
        new_options_lr_args["initial_value"] = prev_options_learning_rate
        options_learning_rate = maybe_make_schedule(new_options_lr_args)

        # set new schedule
        model.meta_pi_clip_range = meta_policy_clip_range
        model.meta_learning_rate = meta_learning_rate
        model.options_pi_clip_range = options_clip_range
        model.options_learning_rate = options_learning_rate

    print(f"Continuing experiment {MODEL_NAME}.")
    print(f"Started {type(model).__name__} training for {remaining_timesteps} steps "
          f"with {n_envs} actors and {n_eval_envs} evaluators...")
    model.learn(total_timesteps=remaining_timesteps,
                callback=cb_list,
                reset_num_timesteps=False, tb_log_name=name)


if __name__ == "__main__":
    run()
