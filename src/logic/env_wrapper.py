from typing import Union, List

import torch as th
from gymnasium import Wrapper, Env, spaces
from ocatari.core import OCAtari
from ocatari.ram.game_objects import GameObject
from nsfr.logic_utils import get_lang

from envs.util import get_env_identifier
from logic.state_extraction import LogicStateExtractor
from logic.base import LARK_PATH, LANG_PATH


class LogicEnvWrapper(Wrapper):
    """Applies the environment-specific logic state extractor to all obs,
    returning logic representations of the states. Also applies reward shaping."""

    def __init__(
        self,
        env: Union[Env, OCAtari],
    ) -> None:
        super().__init__(env)
        self.env_name = get_env_identifier(env.game_name)
        self.logic_state_extractor = LogicStateExtractor.create(self.env_name)
        obs_shape = self.logic_state_extractor.state_shape
        self.observation_space = spaces.Box(-255, 255,
                                            shape=obs_shape,
                                            dtype=self.env.observation_space.dtype)

        self.action_names = env.get_action_meanings()
        self._init_predicates_and_actions()
        self.predicate_space = spaces.Discrete(self.n_predicates)
        self.action_space = self.predicate_space

        self._init_predicates_and_actions()
        self.is_reset = False
        self.ret = 0

    def _init_predicates_and_actions(self):
        """Reads out all predicates from the clauses and determines which predicate
        belongs to which environment action."""

        _, clauses, _, _ = get_lang(LARK_PATH, LANG_PATH, "", self.env_name)

        predicates = []
        pred2action = []
        for clause in clauses:
            if clause.head.pred.name not in predicates:
                predicate_name = clause.head.pred.name
                predicates.append(predicate_name)
                for a, action_name in enumerate(self.action_names):
                    if action_name.lower() in predicate_name:
                        pred2action.append(a)
                        break
                    if a == len(self.action_names):
                        raise ValueError(f"Invalid predicate defined! The predicate '{predicate_name}' does "
                                         f"not contain the name of any action. It must contain any of "
                                         f"'{self.action_names}'.")
        self.predicates = predicates
        self.n_predicates = len(self.predicates)
        self.pred2action = th.tensor(pred2action)

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        self.is_reset = True
        self.ret = 0
        if isinstance(self.env, OCAtari):
            raw_obs = self.env.objects
        logic_obs = self.logic_state_extractor(raw_obs)
        return logic_obs, info

    def step(self, predicate: int):
        action = self.pred2action[predicate]
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(self.env, OCAtari):
            raw_obs = self.env.objects
        logic_obs = self.logic_state_extractor(raw_obs)
        reward += self._get_reward(raw_obs)
        self.ret += reward
        # print(f"\rReturn: {self.ret:.1f}", end="")
        self.is_reset = False
        return logic_obs, reward, terminated, truncated, info

    def _get_reward(self, game_objects: List[GameObject]):
        """Used for convenient reward shaping."""

        if not self.is_reset:
            if self.env_name == 'freeway':
                player = game_objects[0]
                if player.dy < 8:  # ignore situation where player respawns at starting position
                    return -player.dy * 0.025

        return 0
