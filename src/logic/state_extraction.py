from __future__ import annotations
from typing import List
from abc import ABC
import torch as th
import numpy as np
from ocatari.ram.game_objects import GameObject


class LogicStateExtractor(ABC):
    """Turns the raw state of an env into a logic state representation.
    Works only for supported envs."""

    def __init__(self, n_objects: int, n_features: int):
        self.n_objects = n_objects
        self.n_features = n_features

    def __call__(self, obs: List[GameObject]) -> th.Tensor:
        raise NotImplementedError()

    @property
    def state_shape(self):
        return self.n_objects, self.n_features

    @staticmethod
    def create(env_name) -> LogicStateExtractor:
        registered_extractors = {
            "freeway": FreewayExtractor,
            "asterix": AsterixExtractor,
        }
        if env_name not in registered_extractors.keys():
            raise NotImplementedError()
        return registered_extractors[env_name]()


class FreewayExtractor(LogicStateExtractor):
    def __init__(self):
        super().__init__(n_objects=11, n_features=6)

    def __call__(self, obs: List[GameObject]) -> th.Tensor:
        extracted_states = np.zeros((self.n_objects, self.n_features))

        chicken = obs[0]
        cars = obs[2:]

        extracted_states[0][0] = 1
        extracted_states[0][-2:] = chicken.xy

        for i, car in enumerate(cars):
            extracted_states[i+1][1] = 1
            extracted_states[i+1][-2:] = car.xy

        state = th.tensor(extracted_states, dtype=th.float32).unsqueeze(0)
        return state


class AsterixExtractor(LogicStateExtractor):
    def __init__(self):
        super().__init__(n_objects=11, n_features=6)

    def __call__(self, obs: List[GameObject]) -> th.Tensor:
        """Taken from NUDGE."""
        extracted_states = np.zeros((self.n_objects, self.n_features))

        for i, entity in enumerate(obs):
            if entity.category == "Player":
                extracted_states[i][0] = 1
                extracted_states[i][-2:] = entity.xy
            elif entity.category == 'Enemy':
                extracted_states[i][1] = 1
                extracted_states[i][-2:] = entity.xy
            elif "Reward" in entity.category:
                extracted_states[i][2] = 1
                extracted_states[i][-2:] = entity.xy
            else:
                extracted_states[i][3] = 1
                extracted_states[i][-2:] = entity.xy

        state = th.tensor(extracted_states, dtype=th.float32).unsqueeze(0)
        return state
