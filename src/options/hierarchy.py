from typing import List

from gymnasium.spaces import Discrete, Space
from torch import nn

from options.option import Option
from utils.common import get_option_name


class OptionsHierarchy(nn.Module):
    def __init__(self,
                 shape: List[int],
                 observation_space: Space,
                 action_space: Space,
                 lr_schedule,
                 net_arch=None):
        super().__init__()

        if shape is None:
            shape = []
        self.shape = shape
        self.size = len(shape)  # 0 => no options, 1 => one level of options...

        action_option_spaces = []
        for n_options in shape:
            # assert n_options > 1, "It doesn't make sense to have a layer containing only a single option."
            action_option_spaces.append(Discrete(n_options))
        action_option_spaces.append(action_space)
        self.action_option_spaces = action_option_spaces

        def make_option(local_action_space: Space):
            return Option(observation_space=observation_space,
                          action_space=local_action_space,
                          lr_schedule=lr_schedule,
                          net_arch=net_arch)

        self.options = []  # higher-level options first, lower-level options last
        for h, n_options in enumerate(self.shape):
            action_option_space = self.action_option_spaces[h + 1]
            level_options = [make_option(action_option_space) for _ in range(n_options)]
            self.options.append(level_options)

        self._build_attributes()

    def _build_attributes(self):
        """Adds for each option a corresponding attribute to this OptionsHierarchy object.
        These attributes are needed for saving and loading the model."""
        for level_id, level in enumerate(self.options):
            for option_id, option in enumerate(level):
                name = get_option_name(level_id, option_id)
                setattr(self, name, option)

    def __getitem__(self, item) -> List[Option]:
        return self.options[item]
