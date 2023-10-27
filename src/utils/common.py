from __future__ import annotations

from typing import Sequence, Type
import os
import re
from pathlib import Path
import ctypes  # for flashing window in taskbar under Windows
import shutil

import numpy as np
import torch
from ocatari.ram.game_objects import GameObject
from tensorboard import program
from torch import nn

from envs.util import get_atari_identifier


def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs


def get_torch_device(device: str):
    if "cuda" in device and torch.cuda.is_available():
        device = torch.device(device)
    else:
        device = torch.device("cpu")
    return device


def is_sorted(a: np.array) -> bool:
    return np.all(a[:-1] <= a[1:])


def ask_to_override_model(path: Path):
    question = f"There is already a model saved at '{path.as_posix()}'. Override? (y/n)"
    if user_agrees_to(question):
        remove_folder(path)
    else:
        print("No files changed. Shutting down program.")
        quit()


def user_agrees_to(question):
    """Makes a yes/no query to the user. Returns True if user answered yes, False if no, and repeats if
    the user's question was invalid."""
    # let window flash in Windows
    if hasattr(ctypes, "windll"):
        ctypes.windll.user32.FlashWindow(ctypes.windll.kernel32.GetConsoleWindow(), True)

    # ask question to user and handle answer
    while True:
        ans = input(question + "\n")
        if ans == "y":
            return True
        elif ans == "n":
            return False
        else:
            print("Invalid answer. Please type 'y' for 'Yes' or type 'n' for 'No.'")


def remove_folder(path):
    shutil.rmtree(path)


def get_category_counts(objects_categorized: dict[str, Sequence[GameObject]]):
    return {category: len(object_list) for category, object_list in objects_categorized.items()}


def initialize_tensorboard():
    # manual console run: tensorboard --logdir=out/models
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'out/models'])
    url = tb.launch()
    print(f"Tensorboard listening on {url}")


def hyperparams_to_experiment_name(environment_kwargs, seed) -> str:
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


def get_option_name(level: int, position: int) -> str:
    return f"option_{level}_{position}"


def get_most_recent_checkpoint_steps(checkpoint_dir: str | Path) -> int | None:
    checkpoints = os.listdir(checkpoint_dir)
    highest_steps = 0
    pattern = re.compile("[0-9]+")
    for i, c in enumerate(checkpoints):
        match = pattern.search(c)
        if match is not None:
            steps = int(match.group())
            if steps > highest_steps:
                highest_steps = steps
    return highest_steps
