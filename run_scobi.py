from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from utils import maybe_make_schedule, get_experiment_name_from_hyperparams, init_envs, init_callbacks, get_torch_device
from options_ppo import OptionsPPO
from option_critic_policy import OptionCriticPolicy

CONFIG_PATH = "in/config/scobi.yaml"
MODELS_PATH = "out/scobi_sb3/"


def run(name: str = None,
        cuda: bool = True,
        cores: int = 1,
        seed: int = 0,
        environment: dict = None,
        model: dict = None,
        training: dict = None):
    # Set experiment name
    if name is None:
        name = get_experiment_name_from_hyperparams(environment_kwargs=environment, seed=seed)

    device = get_torch_device(cuda)

    object_centric = environment["object_centric"]
    eval_frequency = training["eval_frequency"]
    n_envs = cores
    n_eval_envs = cores
    total_timestamps = int(float(training["total_timesteps"]))
    log_path = Path(MODELS_PATH, name, "logs")
    ckpt_path = Path(MODELS_PATH, name, "checkpoints")
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    train_env, eval_env = init_envs(n_envs=n_envs, n_eval_envs=n_eval_envs, seed=seed, **environment)

    cb_list = init_callbacks(exp_name=name,
                             total_timestamps=total_timestamps,
                             object_centric=object_centric,
                             n_envs=n_envs,
                             eval_env=eval_env,
                             n_eval_episodes=4*n_eval_envs,
                             ckpt_path=ckpt_path,
                             eval_frequency=eval_frequency)

    learning_rate = maybe_make_schedule(training.pop("learning_rate"))
    clip_range = maybe_make_schedule(model["ppo"].pop("clip_range"))

    n_options = model["n_options"]

    if n_options > 1:
        model = OptionsPPO(
            options_policy=OptionCriticPolicy,
            n_options=n_options,
            learning_rate=learning_rate,
            clip_range=clip_range,
            env=train_env,
            **model["ppo"],
            verbose=1,
            device=device)
    else:
        policy = "MlpPolicy" if object_centric else "CnnPolicy"
        model = PPO(
            policy,
            learning_rate=learning_rate,
            clip_range=clip_range,
            env=train_env,
            **model["ppo"],
            verbose=1,
            device=device)

    new_logger = configure(str(log_path), ["tensorboard"])
    model.set_logger(new_logger)

    print(f"Experiment name: {name}")
    print(f"Started {type(model).__name__} training with {n_envs} actors and {n_eval_envs} evaluators...")
    model.learn(total_timesteps=total_timestamps, callback=cb_list)


if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    run(**config)
