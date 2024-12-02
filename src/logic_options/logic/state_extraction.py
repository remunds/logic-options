from __future__ import annotations
from typing import List, Sequence
from abc import ABC
import torch as th
import numpy as np
from ocatari.ram.game_objects import GameObject
from ocatari.ram.seaquest import MAX_ESSENTIAL_OBJECTS


class LogicStateExtractor(ABC):
    """Turns the raw state of an env into a logic state representation.
    Works only for supported envs."""
    obs_dtype = np.int32

    def __init__(self, obs_shape: Sequence[int]):
        self.obs_shape = obs_shape  # the original env obs shape before extraction

    def __call__(self, obs: List[GameObject]) -> th.Tensor:
        raise NotImplementedError()

    @property
    def state_shape(self):
        """The resulting shape after extraction."""
        raise NotImplementedError()

    @staticmethod
    def create(env_name, obs_shape: Sequence[int]) -> LogicStateExtractor:
        registered_extractors = {
            "freeway": FreewayExtractor,
            "asterix": AsterixExtractor,
            "seaquest": SeaquestExtractor,
            "meetingroom": DummyExtractor,
        }
        if env_name not in registered_extractors.keys():
            raise NotImplementedError(f"Logic state extractor for environment '{env_name}' is missing!")
        return registered_extractors[env_name](obs_shape=obs_shape)


class FreewayExtractor(LogicStateExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_objects = 11
        self.n_features = 6

    def __call__(self, obs: List[GameObject]) -> th.Tensor:
        extracted_states = np.zeros((self.n_objects, self.n_features), dtype=self.obs_dtype)

        chicken = obs[0]
        cars = obs[2:-2]

        extracted_states[0][0] = 1
        extracted_states[0][-2:] = chicken.xy

        for i, car in enumerate(cars):
            extracted_states[i+1][1] = 1
            extracted_states[i+1][-2:] = car.xy

        state = th.tensor(extracted_states, dtype=th.float32).unsqueeze(0)
        return state

    @property
    def state_shape(self):
        return self.n_objects, self.n_features


class AsterixExtractor(LogicStateExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_objects = 11
        self.n_features = 6

    def __call__(self, obs: List[GameObject]) -> th.Tensor:
        """Taken from NUDGE."""
        extracted_states = np.zeros((self.n_objects, self.n_features), dtype=self.obs_dtype)

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

    @property
    def state_shape(self):
        return self.n_objects, self.n_features


class SeaquestExtractor(LogicStateExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_objects = 43
        self.n_features = 4  # visible, x-pos, y-pos, right-facing

        # Compute index offsets. Needed to deal with multiple same-category objects
        self.obj_offsets = {}
        offset = 0
        for (obj, max_count) in MAX_ESSENTIAL_OBJECTS.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.relevant_objects = set(MAX_ESSENTIAL_OBJECTS.keys())

    def __call__(self, obs: List[GameObject]) -> th.Tensor:
        state = th.zeros((self.n_objects, self.n_features), dtype=th.int32)

        obj_count = {k: 0 for k in MAX_ESSENTIAL_OBJECTS.keys()}

        for obj in obs:
            if obj.category not in self.relevant_objects:
                continue
            idx = self.obj_offsets[obj.category] + obj_count[obj.category]
            if obj.category == "OxygenBar":
                state[idx] = th.Tensor([1, obj.value, 0, 0])
            else:
                orientation = obj.orientation.value if obj.orientation is not None else 0
                state[idx] = th.tensor([1, *obj.center, orientation])
            obj_count[obj.category] += 1

        return state.unsqueeze(0)

    @property
    def state_shape(self):
        return self.n_objects, self.n_features


class DummyExtractor(LogicStateExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, obs) -> th.Tensor:
        return th.tensor(obs)

    @property
    def state_shape(self):
        return self.obs_shape
