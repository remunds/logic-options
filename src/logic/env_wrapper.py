from typing import Union

from gymnasium import Wrapper, Env, spaces
from logic.state_extraction import LogicStateExtractor
from envs.util import get_env_identifier
from ocatari.core import OCAtari


class LogicEnvWrapper(Wrapper):
    """Applies the environment-specific logic state extractor to all obs,
    returning logic representations of the states."""

    def __init__(
        self,
        env: Union[Env, OCAtari],
    ) -> None:
        super().__init__(env)
        env_identifier = get_env_identifier(env.game_name)
        self.logic_state_extractor = LogicStateExtractor.create(env_identifier)
        obs_shape = self.logic_state_extractor.state_shape
        self.observation_space = spaces.Box(-255, 255,
                                            shape=obs_shape,
                                            dtype=self.env.observation_space.dtype)

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        if isinstance(self.env, OCAtari):
            raw_obs = self.env.objects
        logic_obs = self.logic_state_extractor(raw_obs)
        return logic_obs, info

    def step(self, action: int):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(self.env, OCAtari):
            raw_obs = self.env.objects
        logic_obs = self.logic_state_extractor(raw_obs)
        return logic_obs, reward, terminated, truncated, info
