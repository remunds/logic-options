import argparse
from copy import deepcopy
import time
import signal
import os

import numpy as np
import torch

from option_critic import OptionCriticFeatures, OptionCriticConv
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn
from experience_replay import ReplayBuffer
from utils import make_env, get_torch_device
from logger import Logger


parser = argparse.ArgumentParser(description="Option Critic PyTorch")
parser.add_argument('--env', default='CartPole-v1', help='ROM to run')
parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
parser.add_argument('--frame-skip', default=4, type=int, help='Every how many frames to process')
parser.add_argument('--learning-rate', type=float, default=.0005, help='Learning rate')
parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
parser.add_argument('--epsilon-start', type=float, default=1.0, help=('Starting value for epsilon.'))
parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
parser.add_argument('--epsilon-decay', type=float, default=20000, help=('Number of steps to minimum epsilon.'))
parser.add_argument('--max-history', type=int, default=10000, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes.'))
parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
parser.add_argument('--termination-reg', type=float, default=0.01,
                    help=('Regularization to decrease termination prob.'))
parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy.'))
parser.add_argument('--num-options', type=int, default=2, help=('Number of options to create.'))
parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param.')

parser.add_argument('--max_steps_ep', type=int, default=18000, help='number of maximum steps per episode.')
parser.add_argument('--max_steps_total', type=int, default=int(4e6),
                    help='number of maximum steps to take.')  # bout 4 million
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
parser.add_argument('--seed', type=int, default=0, help='Random seed for numpy, torch, random.')
parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
parser.add_argument('--exp', type=str, default=None, help='optional experiment name')


continue_training = True

MODELS_PATH = "out/models/"


def run(args):
    # Create directories
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    # Setup environment
    env, is_atari = make_env(args.env, args.seed)

    option_critic = OptionCriticConv if is_atari else OptionCriticFeatures
    device = get_torch_device(args.cuda)

    # Create the agent
    option_critic = option_critic(
        in_features=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        num_options=args.num_options,
        temperature=args.temp,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        eps_test=args.optimal_eps,
        device=device
    )
    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)

    optim = torch.optim.RMSprop(option_critic.parameters(), lr=args.learning_rate)

    np.random.seed(args.seed)  # TODO: deprecated
    torch.manual_seed(args.seed)  # TODO: deprecated

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
    run_name = f"{time.strftime('%Y-%m-%d-%H-%M-%S')}-{OptionCriticFeatures.__name__}-{args.env}-{args.exp}"
    logger = Logger(logdir=args.logdir, run_name=run_name)

    transition = 0

    # Setup SIGINT handler
    signal.signal(signal.SIGINT, stop_training)

    # Iterate over episodes
    while transition < args.max_steps_total and continue_training:
        ret = 0  # return (sum of all rewards)
        option_lengths = {opt: [] for opt in range(args.num_options)}

        state, _ = env.reset()
        greedy_option = option_critic.choose_option_greedy(state)
        current_option = 0

        done = False
        episode_length = 0
        terminate_option = True
        current_option_length = 0
        epsilon = None

        # Iterate over transitions
        while not done and episode_length < args.max_steps_ep:
            epsilon = option_critic.epsilon

            if terminate_option:
                option_lengths[current_option].append(current_option_length)
                current_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
                current_option_length = 0

            # Choose action
            action, logp, entropy = option_critic.get_action(state, current_option)

            # Perform transition
            next_state, reward, done, _, _ = env.step(action)

            # Save transition
            buffer.push(state, current_option, reward, next_state, done)
            ret += reward

            # Train?
            actor_loss, critic_loss = None, None
            if len(buffer) > args.batch_size:
                actor_loss = actor_loss_fn(state, current_option, logp, entropy,
                                           reward, done, next_state, option_critic, option_critic_prime, args)
                loss = actor_loss

                if transition % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, args)
                    loss += critic_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                if transition % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            terminate_option, greedy_option = option_critic.predict_option_termination(next_state, current_option)

            # update global steps etc
            transition += 1
            episode_length += 1
            current_option_length += 1
            state = next_state

            logger.log_data(transition, actor_loss, critic_loss, entropy.item(), epsilon)

        logger.log_episode(transition, ret, option_lengths, episode_length, epsilon)

    torch.save(option_critic, MODELS_PATH + "result.pckl")


def stop_training(sig, frame):
    print("Stopping training...")
    global continue_training
    continue_training = False


if __name__ == "__main__":
    args = parser.parse_args()
    run(args)
