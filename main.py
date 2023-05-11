import os

import yaml

from option_critic import OptionCritic
from utils import make_env, get_torch_device

continue_training = True

MODELS_PATH = "out/models/"
CONFIG_PATH = "config/config.yaml"


def run(name: str = None,
        cuda: bool = True,
        seed: int = 0,
        environment: dict = None,
        model: dict = None,
        training: dict = None):
    """

    :param name: The name of this experiment run.
    :param cuda: Whether to use the GPU (with CUDA) or the CPU.
    :param seed: Random number generator initialization seed for reproduction
    :param environment: Environment parameters
    :param model: Model parameters
    :param training: Training parameters
    :return:
    """

    # Create directories
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    # Setup environment
    env, is_atari = make_env(seed=seed, **environment)

    device = get_torch_device(cuda)

    # Create the agent
    option_critic = OptionCritic(
        name=name,
        state_shape=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        device=device,
        is_pixel=is_atari,
        **model
    )

    option_critic.practice(env, **training)


if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    run(**config)
