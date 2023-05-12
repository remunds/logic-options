import gymnasium
import numpy as np
import torch

from gymnasium.wrappers import AtariPreprocessing, TransformReward
from gymnasium.wrappers import FrameStack as FrameStack_

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


def make_env(name, seed, render_mode=None):
    if name == 'fourrooms':
        return Fourrooms(), False

    env = gymnasium.make(name, render_mode=render_mode)
    is_atari = hasattr(gymnasium.envs, 'atari') and isinstance(env.unwrapped, gymnasium.envs.atari.atari_env.AtariEnv)
    if is_atari:
        env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, terminal_on_life_loss=True)
        env = TransformReward(env, lambda r: np.clip(r, -1, 1))
        env = FrameStack(env, 4)

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
