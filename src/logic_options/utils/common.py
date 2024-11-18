from __future__ import annotations

import ctypes  # for flashing window in taskbar under Windows
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Sequence, Type
import yaml

import numpy as np
import torch
from ocatari.ram.game_objects import GameObject
from tensorboard import program
from torch import nn

from logic_options.common import MODELS_BASE_PATH
from logic_options.envs.util import get_atari_identifier, get_env_identifier


def to_model_dir(model_name, env_name):
    env_identifier = get_env_identifier(env_name)
    return Path(MODELS_BASE_PATH, env_identifier, model_name)


def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs


def get_torch_device(device: str):
    if "cuda" in device:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        if device == "cuda":  # pick any free GPU
            gpu = get_free_gpus()[0]
            return torch.device(f"cuda:{gpu}")
        else:  # pick specified GPU
            return torch.device(device)
    else:
        return torch.device("cpu")


def get_free_gpus(threshold_vram_usage=40, threshold_gpu_usage=40, wait=True, sleep_time=10):
    """
    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the VRAM usage is below the
                                              threshold (in percent of the total GPU memory).
        threshold_gpu_usage (int, optional): The maximum usage (in percent) of the GPU to be considered free.
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
        wait (bool, optional): Whether to wait until a GPU is free. Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
    """

    def _scan_for_free_gpus():
        # Get GPU usage information
        gpu_memory = np.array(get_gpu_used_memory())
        gpu_usage = np.array(get_gpu_usage())

        # Keep GPUs under thresholds only
        sufficient_memory = gpu_memory < threshold_vram_usage
        sufficient_idle = gpu_usage < threshold_gpu_usage
        free = sufficient_memory & sufficient_idle
        return np.where(free)[0]

    while True:
        free_gpus = _scan_for_free_gpus()
        if len(free_gpus) > 0 or not wait:
            break
        print(f"No free GPUs found, retrying in {sleep_time}s.")
        time.sleep(sleep_time)

    if len(free_gpus) == 0:
        raise RuntimeError("No free GPUs found.")

    return free_gpus


def get_gpu_used_memory():
    # Use the shell to get GPU memory usage info
    smi_query_result = subprocess.check_output(
        "nvidia-smi --query-gpu=memory.used --format=csv", shell=True
    )
    # Extract the usage information
    gpu_mem_used = smi_query_result.decode("utf-8").split("\n")[1:-1]
    mem_usage = [int(info.split(" ")[0]) for info in gpu_mem_used]

    # Use the shell to get GPU memory usage info
    smi_query_result = subprocess.check_output(
        "nvidia-smi --query-gpu=memory.total --format=csv", shell=True
    )
    # Extract the usage information
    gpu_mem_total = smi_query_result.decode("utf-8").split("\n")[1:-1]
    total_mem = [int(info.split(" ")[0]) for info in gpu_mem_total]

    mem_usage = np.round(np.array(mem_usage) / np.array(total_mem) * 100)

    # Calculate percentage of occupied memory
    print("GPU memory usage (in %):", mem_usage)

    return mem_usage


def get_gpu_usage():
    """Returns a list of integers representing the percentage usage
    of each GPU on the machine."""
    # Use the shell to get GPU usage info
    smi_query_result = subprocess.check_output(
        "nvidia-smi --query-gpu=utilization.gpu --format=csv", shell=True
    )
    # Extract the values we're interested in
    gpu_info = smi_query_result.decode("utf-8").split("\n")[1:-1]
    usage = [int(info.split(" ")[0]) for info in gpu_info]
    print("GPU usage (in %):", usage)
    return usage


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


def update_yaml(file_path, key, value):
    with open(file_path) as f:
        yaml_dict = yaml.safe_load(f)

    yaml_dict[key] = value

    with open(file_path, "w") as f:
        yaml.dump(yaml_dict, f)
