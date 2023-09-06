from __future__ import annotations

from typing import Sequence, Type

import numpy as np
import torch
from ocatari.ram.game_objects import GameObject
from tensorboard import program
from torch import nn

from envs.common import get_atari_identifier


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


def categorize_objects_into_dict(objects: Sequence[GameObject]) -> dict[str, Sequence[GameObject]]:
    """Converts a list of objects into a dict where each key is an object
    category and the value all objects of that category."""
    objects_categorized = {}
    for obj in objects:
        category = type(obj).__name__
        if category not in objects_categorized.keys():
            objects_categorized[category] = [obj]
        else:
            objects_categorized[category].append(obj)
    return objects_categorized


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


def get_option_name(level: int, index: int) -> str:
    return f"option_{level}_{index}"
