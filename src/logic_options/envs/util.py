from __future__ import annotations

from typing import Union

import gymnasium
from ocatari import OCAtari


def get_env_name(env: Union[gymnasium.Env, OCAtari]):
    return env.spec.name if isinstance(env, gymnasium.Env) else env.game_name


def get_env_identifier(env_name: str):
    if "/" in env_name:
        return get_atari_identifier(env_name)
    else:
        return env_name.lower()


def get_atari_identifier(env_name: str):
    """Extracts game name, e.g.: 'ALE/Pong-v5' => 'pong'"""
    return env_name.split("/")[1].split("-")[0].lower()
