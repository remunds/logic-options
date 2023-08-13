from __future__ import annotations

import os
from typing import Sequence, Union, Callable, Type

import gymnasium
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import AtariPreprocessing, TransformReward
from gymnasium.wrappers import FrameStack  # as FrameStack_
from ocatari.core import OCAtari
from ocatari.ram.game_objects import GameObject
from scobi import Environment as ScobiEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from tensorboard import program
from torch import nn

from fourrooms import Fourrooms

ATARI_SCREEN_WIDTH = 160
ATARI_SCREEN_HEIGHT = 210
AVG_SCREEN_VELOCITY = 4  # TODO: test
FOCUS_FILES_DIR = "in/focusfiles"
FOCUS_FILES_DIR_EXTERNAL = "in/focusfiles_external"


def make_env(name, seed, render_mode=None, framestack=4,
             frameskip=None, object_centric=False, **kwargs) -> (gymnasium.Env, bool):
    if name == "fourrooms":
        if object_centric:
            raise NotImplemented("Four Rooms not implemented yet as object centric.")
        else:
            return Fourrooms(), False

    is_atari = 'ALE' in name

    gym_frameskip = 1 if is_atari else frameskip

    # Get environment object
    if object_centric:
        if not is_atari:
            raise NotImplemented("Object centric for non-Atari games not implemented yet.")
        env_name = name.split("/")[1].split("-")[0]  # Extract environment name from gymnasium env identifier
        env = OCAtari(env_name, mode='revised', hud=False, render_mode=render_mode,
                      frameskip=gym_frameskip, **kwargs)
    else:
        env = gymnasium.make(name, render_mode=render_mode, frameskip=gym_frameskip, **kwargs)

    # if is_atari:
    frameskip = 4 if frameskip is None else frameskip
    env = AtariPreprocessing(env,
                             grayscale_obs=True,
                             scale_obs=True,
                             terminal_on_life_loss=True,
                             frame_skip=frameskip)
    env = TransformReward(env, lambda r: float(np.clip(r, -1, 1)))
    env = FrameStack(env, framestack)

    # Initialize environment using seed
    env.reset(seed=seed, options={})

    return env, is_atari


def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs


def get_torch_device(use_cuda: bool = True):
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    if device.type == 'cuda':
        print("Using GPU")
    else:
        print("Using CPU")
    return device


def is_sorted(a: np.array) -> bool:
    return np.all(a[:-1] <= a[1:])


def num2text(num):
    if num == 0:
        return "0"
    elif np.abs(num) < 1:
        return "%.2f" % num
    elif np.abs(num) < 10 and num % 1 != 0:
        return "%.1f" % num
    elif np.abs(num) < 1000:
        return "%.0f" % num
    elif np.abs(num) < 10000:
        thousands = num / 1000
        return "%.1fK" % thousands
    elif np.abs(num) < 1e6:
        thousands = num / 1000
        return "%.0fK" % thousands
    elif np.abs(num) < 1e7:
        millions = num / 1e6
        return "%.1fM" % millions
    else:
        millions = num / 1e6
        return "%.0fM" % millions


def sec2hhmmss(s):
    m = s // 60
    h = m // 60
    return "%d:%02d:%02d h" % (h, m % 60, s % 60)


def objects_to_matrix(objects: Sequence[GameObject]):
    """Converts a list of objects into the corresponding matrix representation.
    Each row (x, y, dx, dy) of the returned matrix represents the object's center (!)
    position and velocity vector."""

    object_matrix = np.zeros(shape=(len(objects), 4), dtype='float32')
    object_matrix[:] = np.nan

    for i, obj in enumerate(objects):
        if obj is not None:
            object_matrix[i] = (obj.x + obj.w / 2,
                                obj.y + obj.h / 2,
                                obj.dx,
                                obj.dy)

    return object_matrix


def normalize_object_matrix(object_matrix):
    # Normalize positions to [-1, 1] TODO: this applies only to Atari => generalize
    object_matrix[:, 0] = 2 * object_matrix[:, 0] / ATARI_SCREEN_WIDTH - 1
    object_matrix[:, 1] = 2 * object_matrix[:, 1] / ATARI_SCREEN_HEIGHT - 1

    # Normalize velocities with tanh to [-1, 1]
    object_matrix[:, 2:4] = np.tanh(object_matrix[:, 2:4] / AVG_SCREEN_VELOCITY)

    # Replace NaNs with zeros
    np.nan_to_num(object_matrix, nan=0, copy=False)

    return object_matrix


def categorize_objects_into_dict(objects: Sequence[GameObject]):
    objects_categorized = {}
    for obj in objects:
        category = type(obj).__name__
        if category not in objects_categorized.keys():
            objects_categorized[category] = [obj]
        else:
            objects_categorized[category].append(obj)
    return objects_categorized


def get_category_counts(objects_categorized: dict['str', Sequence[GameObject]]):
    return {category: len(object_list) for category, object_list in objects_categorized.items()}


def pad_object_list(objects: Sequence[GameObject], max_object_counts: dict):
    """Takes a list of objects and fills it with None entries where other
    objects are missing (compared to max_object_counts)."""
    padded_object_list = []
    objects_categorized = categorize_objects_into_dict(objects)

    # Construct padded object list iteratively by category
    for category in list(max_object_counts.keys()):
        if category in objects_categorized.keys():
            category_objects = objects_categorized[category]
            padded_object_list.extend(category_objects)
        else:
            category_objects = []

        # Add padding for this category
        padding_length = max_object_counts[category] - len(category_objects)
        padded_object_list.extend(padding_length * [None])

    return padded_object_list


def initialize_tensorboard():
    # manual console run: tensorboard --logdir=out/models
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'out/models'])
    url = tb.launch()
    print(f"Tensorboard listening on {url}")


def get_env_name(env: Union[gymnasium.Env, OCAtari]):
    return env.spec.name if isinstance(env, gymnasium.Env) else env.game_name


def get_linear_schedule(initial_value: float) -> Callable[[float], float]:
    def linear(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return linear


def get_exponential_schedule(initial_value: float, half_life_period: float = 0.25) -> Callable[[float], float]:
    """It holds exponential(half_life_period) = 0.5. If half_life_period == 0.25, then
    exponential(0) ~= 0.06"""
    assert 0 < half_life_period < 1

    def exponential(progress_remaining: float) -> float:
        return initial_value * np.exp((1 - progress_remaining) * np.log(0.5) / half_life_period)

    return exponential


def maybe_make_schedule(args):
    if isinstance(args, (int, float)):
        return args
    elif isinstance(args, dict):
        schedule_type = args.pop("schedule_type")
        if schedule_type == "linear":
            return get_linear_schedule(**args)
        if schedule_type == "exponential":
            return get_exponential_schedule(**args)
        elif schedule_type is None:
            return args["initial_value"]
        else:
            ValueError(f"Unrecognized schedule type {schedule_type} provided.")
    else:
        raise ValueError("Invalid parameter schedule specification.")


def get_experiment_name_from_hyperparams(environment_kwargs, seed):
    # Build string from settings hyperparams
    settings_str = ""

    reward_mode = environment_kwargs['reward_mode']

    if reward_mode == "env":
        settings_str += "_re"
    elif reward_mode == "human":
        settings_str += "_rh"
    elif reward_mode == "mixed":
        settings_str += "_rm"

    if environment_kwargs["prune_concept"] == "default":
        settings_str += "_pr-d"
    elif environment_kwargs["prune_concept"] == "external":
        settings_str += "_pr-e"

    if environment_kwargs["exclude_properties"]:
        settings_str += '_ep'

    game_name = get_atari_identifier(environment_kwargs["name"])

    exp_name = game_name + "_s" + str(seed) + settings_str

    return exp_name


def get_atari_identifier(env_name: str):
    """Extracts game name, e.g.: 'ALE/Pong-v5' => 'pong'"""
    return env_name.split("/")[1].split("-")[0].lower()


REWARD_MODE = {
    "env": 0,
    "human": 1,
    "mixed": 2
}
MULTIPROCESSING_START_METHOD = "spawn" if os.name == 'nt' else "fork"  # 'nt' == Windows


def init_envs(name: str,
              n_envs: int,
              n_eval_envs: int,
              prune_concept: str,
              exclude_properties: bool,
              reward_mode: str,
              seed: int,
              object_centric: bool = True,
              frameskip: int = 4,
              framestack: int = 1,
              eval_render: bool = False) -> (SubprocVecEnv, SubprocVecEnv):
    eval_env_seed = (seed + 42) * 2  # different seeds for eval

    if object_centric:
        if prune_concept == "default":
            focus_dir = FOCUS_FILES_DIR
        elif prune_concept == "external":
            focus_dir = FOCUS_FILES_DIR_EXTERNAL
        else:
            raise ValueError(f"Unknown prune concept '{prune_concept}'.")

        # Extract game name if Atari
        pruned_ff_name = get_pruned_focus_file_from_env_name(name)

        # Verify compatibility with Gymnasium
        monitor = make_scobi_env(name=name,
                                 focus_dir=focus_dir,
                                 pruned_ff_name=pruned_ff_name,
                                 exclude_properties=exclude_properties,
                                 reward_mode=REWARD_MODE[reward_mode])()
        check_env(monitor.env)
        del monitor

        eval_render_mode = "human" if eval_render else None

        # silent init and don't refresh default yaml file because it causes spam and issues with multiprocessing
        train_envs = [make_scobi_env(name=name,
                                     focus_dir=focus_dir,
                                     pruned_ff_name=pruned_ff_name,
                                     exclude_properties=exclude_properties,
                                     rank=i,
                                     seed=seed,
                                     silent=True,
                                     refresh=False,
                                     reward_mode=REWARD_MODE[reward_mode]) for i in range(n_envs)]
        eval_envs = [make_scobi_env(name=name,
                                    focus_dir=focus_dir,
                                    pruned_ff_name=pruned_ff_name,
                                    exclude_properties=exclude_properties,
                                    rank=i,
                                    seed=eval_env_seed,
                                    silent=True,
                                    refresh=False,
                                    reward_mode=0,
                                    render_mode=eval_render_mode) for i in range(n_eval_envs)]

        env = SubprocVecEnv(train_envs, start_method=MULTIPROCESSING_START_METHOD)
        eval_env = SubprocVecEnv(eval_envs, start_method=MULTIPROCESSING_START_METHOD)

        return env, eval_env

    else:
        env = make_atari_env(name,
                             n_envs=4,
                             seed=seed,
                             vec_env_cls=SubprocVecEnv,
                             vec_env_kwargs={"start_method": MULTIPROCESSING_START_METHOD},
                             wrapper_kwargs={"frame_skip": frameskip})
        env = VecFrameStack(env, n_stack=framestack)

        eval_env = make_atari_env(name,
                                  n_envs=4,
                                  seed=eval_env_seed,
                                  vec_env_cls=SubprocVecEnv,
                                  vec_env_kwargs={"start_method": MULTIPROCESSING_START_METHOD},
                                  wrapper_kwargs={"frame_skip": frameskip})
        eval_env = VecFrameStack(eval_env, n_stack=framestack)

        return env, eval_env


def get_pruned_focus_file_from_env_name(name: str) -> str:
    if "ALE" in name:
        env_identifier = get_atari_identifier(name)
    else:
        env_identifier = name
    return f"{env_identifier}.yaml"


def make_scobi_env(name: str,
                   focus_dir: str,
                   pruned_ff_name: str,
                   exclude_properties: bool,
                   rank: int = 0,
                   seed: int = 0,
                   silent=False,
                   refresh=True,
                   reward_mode=0,
                   **kwargs) -> Callable:
    def _init() -> gym.Env:
        env = ScobiEnv(name,
                       focus_dir=focus_dir,
                       focus_file=pruned_ff_name,
                       hide_properties=exclude_properties,
                       silent=silent,
                       reward=reward_mode,
                       refresh_yaml=refresh,
                       hud=True,
                       **kwargs)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def get_net_from_layer_dims(layers_dims: list[int],
                            in_dim: int,
                            activation_fn: Type[nn.Module]) -> (list[nn.Module], int):
    net: list[nn.Module] = []
    last_layer_dim = in_dim
    for layer_dim in layers_dims:
        net.append(nn.Linear(last_layer_dim, layer_dim))
        net.append(activation_fn())
        last_layer_dim = layer_dim
    return net, last_layer_dim


def get_option_name(level: int, index: int) -> str:
    return f"option_{level}_{index}"
