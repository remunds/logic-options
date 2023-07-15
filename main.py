import os
import shutil

import yaml

from agent import Agent, get_model_dir
from option_critic import OptionCritic
from utils import make_env, get_torch_device, get_env_name


continue_training = True

CONFIG_PATH = "config/config.yaml"
MODELS_PATH = "out/models/"


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

    # Setup environment and computing device
    env, is_atari = make_env(seed=seed, **environment)
    env_name = get_env_name(env)

    device = get_torch_device(cuda)

    model_dir = get_model_dir(env_name=env_name, model_name=name)

    # Handle any pre-existing models
    load_existing_agent = False
    if os.path.exists(model_dir):
        ans = None
        while ans not in ['l', 'r', 'q']:
            ans = input(f"There already exists a model at '{model_dir}'. You can either load (l) or "
                        f"remove (r) the model. Else, you can quit (q) instead.")
        match ans:
            case 'l':  # load
                load_existing_agent = True
            case 'r':  # remove
                os.rmdir(model_dir)
                os.makedirs(model_dir)
            case 'q':
                quit()

    if load_existing_agent:
        agent = Agent.load(env_name=env_name, model_name=name)
        # TODO: copy train config
    else:  # Create the agent and copy the config
        option_critic = OptionCritic(
            env=env,
            device=device,
            is_object_centric=environment['object_centric'],
            **model
        )
        agent = Agent(name=name, model=option_critic, env=env, is_object_centric=environment['object_centric'])
        shutil.copy(src=CONFIG_PATH, dst=agent.model_dir)

    agent.practice(env, **training)


if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    run(**config)
