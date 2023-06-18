from __future__ import annotations

import gymnasium
import numpy as np
import torch
from typing import Sequence

from ocatari.core import OCAtari
from ocatari.ram.game_objects import GameObject
from gymnasium.wrappers import AtariPreprocessing, TransformReward
from gymnasium.wrappers import FrameStack as FrameStack_
from tensorboard import program

from fourrooms import Fourrooms


class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


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
        env = OCAtari(env_name, mode='revised', hud=True, render_mode=render_mode,
                      frameskip=gym_frameskip, **kwargs)
    else:
        env = gymnasium.make(name, render_mode=render_mode, frameskip=gym_frameskip, **kwargs)

    if is_atari:
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


def objects_to_numeric_vector(objects: Sequence[GameObject]):
    """Converts a list of objects into the corresponding matrix representation.
    Each row (x, y, dx, dy) of the returned matrix represents the object's center (!)
    position and velocity vector."""
    object_matrix = np.zeros(shape=(len(objects), 4), dtype='float32')
    for i, obj in enumerate(objects):
        if obj is not None:
            object_matrix[i] = (obj.x + obj.w // 2,
                                obj.y + obj.h // 2,
                                obj.dx,
                                obj.dy)
    return object_matrix.flatten()


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
    padded_object_list = []
    objects_categorized = categorize_objects_into_dict(objects)

    # Construct padded object list iteratively by category
    for category in list(max_object_counts.keys()):
        category_objects = objects_categorized[category]
        padded_object_list.extend(category_objects)

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
