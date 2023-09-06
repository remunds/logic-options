from __future__ import annotations

import os
from typing import Union, Callable

import gymnasium
import gymnasium as gym
from ocatari import OCAtari
from scobi import Environment as ScobiEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from common import FOCUS_FILES_DIR, FOCUS_FILES_DIR_EXTERNAL, REWARD_MODE


def get_env_name(env: Union[gymnasium.Env, OCAtari]):
    return env.spec.name if isinstance(env, gymnasium.Env) else env.game_name


def get_atari_identifier(env_name: str):
    """Extracts game name, e.g.: 'ALE/Pong-v5' => 'pong'"""
    return env_name.split("/")[1].split("-")[0].lower()


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


MULTIPROCESSING_START_METHOD = "spawn" if os.name == 'nt' else "fork"  # 'nt' == Windows


def init_envs(name: str,
              n_envs: int,
              n_eval_envs: int,
              seed: int,
              prune_concept: str = None,
              exclude_properties: bool = None,
              reward_mode: str = None,
              object_centric: bool = True,
              frameskip: int = 4,
              framestack: int = 1,
              eval_render: bool = False,
              normalize: bool = True,
              freeze_invisible_obj: bool = False) -> (SubprocVecEnv, SubprocVecEnv):
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
                                 reward_mode=REWARD_MODE[reward_mode],
                                 normalize=normalize,
                                 freeze_invisible_obj=freeze_invisible_obj)()
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
                                     reward_mode=REWARD_MODE[reward_mode],
                                     normalize=normalize,
                                     freeze_invisible_obj=freeze_invisible_obj) for i in range(n_envs)]
        eval_envs = [make_scobi_env(name=name,
                                    focus_dir=focus_dir,
                                    pruned_ff_name=pruned_ff_name,
                                    exclude_properties=exclude_properties,
                                    rank=i,
                                    seed=eval_env_seed,
                                    silent=True,
                                    refresh=False,
                                    reward_mode=0,
                                    render_mode=eval_render_mode,
                                    normalize=normalize,
                                    freeze_invisible_obj=freeze_invisible_obj) for i in range(n_eval_envs)]

        env = SubprocVecEnv(train_envs, start_method=MULTIPROCESSING_START_METHOD)
        eval_env = SubprocVecEnv(eval_envs, start_method=MULTIPROCESSING_START_METHOD)

        return env, eval_env

    else:
        env = make_atari_env(name,
                             n_envs=n_envs,
                             seed=seed,
                             vec_env_cls=SubprocVecEnv,
                             vec_env_kwargs={"start_method": MULTIPROCESSING_START_METHOD},
                             wrapper_kwargs={"frame_skip": frameskip})
        env = VecFrameStack(env, n_stack=framestack)

        eval_env = make_atari_env(name,
                                  n_envs=n_eval_envs,
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
