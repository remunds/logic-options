from __future__ import annotations

from typing import Callable
from pathlib import Path

import gymnasium as gym
from ocatari import OCAtari
from scobi import Environment as ScobiEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, VecFrameStack, VecNormalize, DummyVecEnv

from envs.meeting_room import MeetingRoom
from logic.env_wrapper import LogicEnvWrapper

from common import FOCUS_FILES_DIR, FOCUS_FILES_DIR_UNPRUNED, REWARD_MODE, MULTIPROCESSING_START_METHOD
from envs.util import get_atari_identifier


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
                       reward_mode=reward_mode,
                       refresh_yaml=refresh,
                       hud=True,
                       **kwargs)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def make_logic_env(name: str, seed: int = None, **kwargs):
    def _init():
        if name == "MeetingRoom":
            raw_env = MeetingRoom(**kwargs)
        elif "ALE" in name:
            raw_env = OCAtari(name, mode="revised", hud=False, **kwargs)
        else:
            raise NotImplementedError()
        raw_env.reset(seed=seed)
        return Monitor(LogicEnvWrapper(raw_env))
    return _init


def init_train_eval_envs(n_train_envs: int,
                         n_eval_envs: int,
                         seed: int,
                         reward_mode: str = None,
                         render_eval: bool = False,
                         **kwargs) -> (VecEnv, VecEnv):
    eval_seed = (seed + 42) * 2  # different seeds for eval
    train_env = init_vec_env(n_envs=n_train_envs,
                             seed=seed,
                             train=True,
                             reward_mode=reward_mode,
                             **kwargs)
    eval_render_mode = "human" if render_eval else None
    eval_env = init_vec_env(n_envs=n_eval_envs,
                            seed=eval_seed,
                            train=False,
                            reward_mode="env",
                            render_mode=eval_render_mode,
                            **kwargs)
    return train_env, eval_env


def init_vec_env(name: str,
                 n_envs: int,
                 seed: int,
                 logic: bool = False,
                 object_centric: bool = False,
                 reward_mode: str = None,
                 prune_concept: str = None,
                 exclude_properties: bool = None,
                 frameskip: int = 4,
                 framestack: int = 1,
                 normalize_observation: bool = False,
                 normalize_reward: bool = False,
                 vec_norm_path: str = None,
                 train: bool = False,
                 freeze_invisible_obj: bool = False,
                 render_mode: str = None,
                 render_oc_overlay: bool = False,
                 settings: dict = None) -> VecEnv | None:
    """Helper function to initialize a vector environment with specified parameters."""

    if n_envs == 0:
        return None

    if settings is None:
        settings = dict()
    settings["render_mode"] = render_mode

    if logic:
        assert n_envs == 1
        assert framestack == 1
        assert not normalize_observation
        vec_env = DummyVecEnv([make_logic_env(name, **settings)])

    elif name == "MeetingRoom":
        if n_envs > 1:
            vec_env_cls = SubprocVecEnv
            vec_env_kwargs = {"start_method": MULTIPROCESSING_START_METHOD}
        else:
            vec_env_cls = DummyVecEnv
            vec_env_kwargs = None
        vec_env = make_vec_env(MeetingRoom,
                               n_envs=n_envs,
                               seed=seed,
                               env_kwargs=settings,
                               vec_env_cls=vec_env_cls,
                               vec_env_kwargs=vec_env_kwargs)
        vec_env = VecFrameStack(vec_env, n_stack=framestack)

    elif object_centric:
        if prune_concept == "unpruned":
            focus_dir = FOCUS_FILES_DIR_UNPRUNED
            pruned_ff_name = None
        elif prune_concept == "default":
            focus_dir = FOCUS_FILES_DIR
            pruned_ff_name = get_pruned_focus_file_from_env_name(name)
        else:
            raise ValueError(f"Unknown prune concept '{prune_concept}'.")

        reward_mode = REWARD_MODE[reward_mode]

        # Verify compatibility with Gymnasium and refresh focus YAML file
        monitor = make_scobi_env(name=name,
                                 focus_dir=focus_dir,
                                 pruned_ff_name=pruned_ff_name,
                                 exclude_properties=exclude_properties,
                                 reward_mode=reward_mode,
                                 freeze_invisible_obj=freeze_invisible_obj)()
        check_env(monitor.env)
        del monitor

        envs = [make_scobi_env(name=name,
                               focus_dir=focus_dir,
                               pruned_ff_name=pruned_ff_name,
                               exclude_properties=exclude_properties,
                               rank=i,
                               seed=seed,
                               silent=True,
                               refresh=False,
                               reward_mode=reward_mode,
                               render_mode=render_mode,
                               freeze_invisible_obj=freeze_invisible_obj,
                               render_oc_overlay=render_oc_overlay) for i in range(n_envs)]

        if n_envs > 1:
            vec_env = SubprocVecEnv(envs, start_method=MULTIPROCESSING_START_METHOD)
        else:
            vec_env = DummyVecEnv(envs)

    else:
        if n_envs > 1:
            vec_env_cls = SubprocVecEnv
            vec_env_kwargs = {"start_method": MULTIPROCESSING_START_METHOD}
        else:
            vec_env_cls = DummyVecEnv
            vec_env_kwargs = None
        vec_env = make_atari_env(name,
                                 n_envs=n_envs,
                                 seed=seed,
                                 env_kwargs=settings,
                                 vec_env_cls=vec_env_cls,
                                 vec_env_kwargs=vec_env_kwargs,
                                 wrapper_kwargs={"frame_skip": frameskip})
        vec_env = VecFrameStack(vec_env, n_stack=framestack)

    # Wrap with (either existing or new) VecNormalize to normalize obs and/or reward
    if vec_norm_path is not None:
        env = VecNormalize.load(vec_norm_path, vec_env)
        env.training = train
    else:
        env = VecNormalize(vec_env,
                           norm_obs=normalize_observation,
                           norm_reward=normalize_reward,
                           training=train)

    return env


def get_pruned_focus_file_from_env_name(name: str) -> str:
    if "ALE" in name:
        env_identifier = get_atari_identifier(name)
    else:
        env_identifier = name
    return f"{env_identifier}.yaml"


def get_focus_file_path(prune_concept: str, env_name: str) -> Path:
    if "ALE" in env_name:
        env_identifier = get_atari_identifier(env_name)
    else:
        env_identifier = env_name

    if prune_concept == "default":
        return Path(FOCUS_FILES_DIR, f"{env_identifier}.yaml")
    elif prune_concept == "unpruned":
        return Path(FOCUS_FILES_DIR_UNPRUNED, f"default_focus_{env_name[4:]}.yaml")
    else:
        raise ValueError(f"Unknown prune concept {prune_concept} given.")
