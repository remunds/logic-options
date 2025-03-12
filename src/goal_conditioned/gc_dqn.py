# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    EpisodicLifeEnv,
    FireResetEnv,
    NoopResetEnv,
)
# from stable_baselines3.common.buffers import ReplayBuffer
from buffer import MultiRewardReplayBuffer 
from torch.utils.tensorboard import SummaryWriter

from logic_options.utils.train_monitor import TrainMonitor
from logic_options.utils.normalize_obs_torch import RunningMeanStd
from rtpt import RTPT

import importlib
import sys
import yaml


@dataclass
class Args:
    exp_name: str = "HER" 
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "cuda:13"
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "ALE/Seaquest-v5"
    """the id of the environment"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    max_env_steps: int = 10_000
    """max timesteps of the environment"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.1
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80_000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""

    norm_obs: bool = True
    """if toggled, the observations will be normalized (my implementation)"""

    save_model_steps: int = 1_000_000
    """number of steps between saving the model"""
    checkpoint_path: str = None 
    """path to the checkpoint file to resume training from"""
    neural_meta_policy: bool = False
    """if toggled, a neural meta policy will be used, otherwise a function will be used"""
    atari_wrappers: bool = False
    """if toggled, the atari wrappers will be used"""
    # meta_policy_path: str = None
    meta_policy_path = "in/logic/llm/seaquest-meta-policy.py"
    """path to the meta policy function"""
    num_subpolicies: int = 3
    """number of subpolicies"""
    # rewardfunc_path: str = None 
    backend: str = "HackAtari"
    """the backend to use (HackAtari, OCAtari, Gym)"""
    modifs: str = ""
    """the modifications to apply to the environment"""
    buffer_window_size: int = 4
    """the buffer window size"""
    obs_mode: str = "obj"
    """the observation mode"""
    frameskip: int = 4
    """the number of frames to skip"""
    hud: bool = True
    """if toggled, the HUD will be used"""
    rewardfunc_path = ["in/reward_funcs/seaquest/hud/fight_enemies.py",
                        "in/reward_funcs/seaquest/hud/collect_divers.py",
                        "in/reward_funcs/seaquest/hud/surface.py",
                       ]
    # rewardfunc_path = "in/reward_funcs/seaquest/hud/hackatari_reward.py"
    """path to the reward function(s)"""
    args_file: str = None #
    """path to the args file to load arguments from"""

# Function to create a gym environment with the specified settings
def make_env(env_id, idx, capture_video, run_dir):
    """
    Creates a gym environment with the specified settings.
    """
    def thunk():
        # Setup environment based on backend type (HackAtari, OCAtari, Gym)
        if args.backend == "HackAtari":
            from hackatari.core import HackAtari
            modifs = [i for i in args.modifs.split(" ") if i]
            env = HackAtari(
                env_id,
                modifs=modifs,
                rewardfunc_path=args.rewardfunc_path,
                obs_mode=args.obs_mode,
                hud=args.hud,
                render_mode="rgb_array",
                frameskip=args.frameskip
            )
        elif args.backend == "OCAtari":
            from ocatari.core import OCAtari
            env = OCAtari(
                env_id,
                hud=args.hud,
                render_mode="rgb_array",
                obs_mode=args.obs_mode,
                frameskip=args.frameskip
            )
        elif args.backend == "Gym":
            # Use Gym backend with image preprocessing wrappers
            env = gym.make(env_id, render_mode="rgb_array", frameskip=args.frameskip)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, args.buffer_window_size)
        else:
            raise ValueError("Unknown Backend")

        # Capture video if required
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env,
                                           f"{run_dir}/media/videos",
                                           disable_logger=True)

        # Apply standard Atari environment wrappers
        env = TrainMonitor(env)
        env = NoopResetEnv(env, noop_max=30)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        return env

    return thunk

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, envs, device, norm_obs=False, meta=False, num_subpolicies=None):
        super().__init__()
        self.meta = meta
        if meta and num_subpolicies is None:
            raise ValueError("num_subpolicies must be specified if meta=True")
        self.network = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_subpolicies if meta else envs.single_action_space.n) 
        )
        self.norm_obs = norm_obs
        if norm_obs:
            shape = (np.prod(envs.single_observation_space.shape),)
            self.obs_rms = RunningMeanStd(shape=shape, device=device)
            self.epsilon = 1e-8

    def _rms_normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        # only update if batch size is > 1
        with torch.no_grad():
            if obs.shape[0] > 1:
                self.obs_rms.update(obs)
            new_obs = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + self.epsilon)
        return new_obs.to(torch.float32)

    def forward(self, x):
        if self.norm_obs:
            x = self._rms_normalize(x)
        return self.network(x)
    
def save(save_path, q_nets, optimizers, global_step, args):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
    torch.save({
        'global_step': global_step,
        'model_state_dict': [q_net.state_dict() for q_net in q_nets],
        'optimizer_state_dict': [optimizer.state_dict() for optimizer in optimizers],
        'args': args.__dict__,
    }, save_path)

def save_args(args, save_path):
    with open(save_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

def load_args(args_file):
    with open(args_file, 'r') as f:
        args_dict = yaml.safe_load(f)
    return Args(**args_dict)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    # assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.args_file:
        print(f"Loading arguments from {args.args_file}")
        args = load_args(args.args_file)
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    model_save_dir = f"runs/{run_name}/models"
    os.makedirs(model_save_dir, exist_ok=True)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    save_args(args, f"runs/{run_name}/args.yaml")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device)
    n_rewards = len(args.rewardfunc_path) if args.rewardfunc_path is not None else 1

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        # [make_hackatari_env(args.env_id, i, **hackatari_args) for i in range(args.num_envs)],
        [make_env(args.env_id, i, args.capture_video, f"runs/{run_name}") for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_networks = [QNetwork(envs, device, norm_obs=args.norm_obs).to(device) for _ in range(args.num_subpolicies)]
    target_networks = [QNetwork(envs, device, norm_obs=args.norm_obs).to(device) for _ in range(args.num_subpolicies)]
    for i, target_network in enumerate(target_networks):
        target_network.load_state_dict(q_networks[i].state_dict())

    # neural meta policy
    if args.neural_meta_policy:
        meta_policy_q = QNetwork(envs, device, norm_obs=False, meta=True, num_subpolicies=args.num_subpolicies).to(device)
        meta_target = QNetwork(envs, device, norm_obs=False, meta=True, num_subpolicies=args.num_subpolicies).to(device)
        meta_target.load_state_dict(meta_policy_q.state_dict())
        q_networks.append(meta_policy_q)
        target_networks.append(meta_target)
        meta_policy_func = None

    # load meta-policy function
    elif args.meta_policy_path is not None:
        spec = importlib.util.spec_from_file_location("meta_policy", args.meta_policy_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["meta_policy"] = module
        spec.loader.exec_module(module)
        meta_policy_func = module.meta_policy
        meta_policy_q = None
        meta_target = None
    
    else:
        raise ValueError("Either neural_meta_policy or meta_policy_path must be specified")

    optimizers = [optim.Adam(q_network.parameters(), lr=args.learning_rate) for q_network in q_networks]

    rb = MultiRewardReplayBuffer(
        n_rewards,
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    highest_episodic_return = float('-inf')
    obs, _ = envs.reset(seed=args.seed)

    curr_episode_rewards = np.zeros((args.max_env_steps, args.num_envs, n_rewards))
    curr_episode_choices = np.ones((args.max_env_steps, args.num_envs), dtype=int) * -1

    rtpt = RTPT(name_initials='RE', experiment_name='gc_dqn', max_iterations=args.total_timesteps)
    rtpt.start()
    # for global_step in range(args.total_timesteps):
    episode_step = 0
    for global_step in range(0, args.total_timesteps, args.num_envs):
        episode_step += 1
        rtpt.step()
        # global_step
        # ALGO LOGIC: put action logic here
        #TODO: torch.no_grad()?
        if meta_policy_q is not None:
            q_values = meta_policy_q(torch.Tensor(obs).to(device))
            option_choices = torch.argmax(q_values, dim=1).cpu().numpy()
        elif meta_policy_func is not None:
            option_choices = meta_policy_func(obs)
        else:
            raise ValueError("Either neural_meta_policy or meta_policy_path must be specified")
        # option_choices now contains the currently active subpolicy for each env
        curr_episode_choices[episode_step] = option_choices
        # obs is shape (num_envs, obs_dim)

        # greedy action (using the current Q-network)
        # for us this means: select subpolicy (sub-q_network) greedily using meta-policy (meta-q_network/func)
        # then select action greedily using the subpolicy
        #TODO: this could be adapted to allow exploration of subpolicies even if they become active towards the end of the training (e.g. because their precondition was not met for a long time)
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            # exploration
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = np.zeros(envs.num_envs)
            for i, q_net in enumerate(q_networks):
                # get obs of envs, where current subpolicy is active
                l_obs = obs[option_choices == i] 
                if l_obs.shape[0] > 0:
                    q_values = q_net(torch.Tensor(l_obs).view(l_obs.shape[0], -1).to(device))
                    l_actions = torch.argmax(q_values, dim=1).cpu().numpy()
                    actions[option_choices == i] = l_actions


        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions.astype(np.int32))

        if "all_rewards" in infos:
            rewards = np.array([np.array(a_r) if a_r is not None else np.array([0.0 for _ in range(n_rewards)]) for a_r in infos["all_rewards"]])
        elif "final_info" in infos and "all_rewards" in infos["final_info"][0]:
            # makes sure that we do not use all_rewards from train_monitor (which is actually return if finished)
            all_rewards = np.zeros((args.num_envs, n_rewards))
            for i, info in enumerate(infos["final_info"]):
                if info and "all_rewards" in info:
                    all_rewards[i] = info["all_rewards"]
            rewards = all_rewards 
        else:
            raise ValueError("all_rewards not in infos")
        
        curr_episode_rewards[episode_step] = rewards

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episodic_return = info["episode"]["r"]
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    if episodic_return > highest_episodic_return:
                        highest_episodic_return = episodic_return
                        save_path = f"{model_save_dir}/best_return.pt"
                        save(save_path, q_networks, optimizers, global_step, args)
                        print(f"New highest episodic return: {episodic_return}. Model saved to {save_path}")
                    if "all_rewards" in info["episode"] and isinstance(info["episode"]["all_rewards"], list):
                        all_rewards = info["episode"]["all_rewards"]

                        # these are the returns 'as if subpolicies were always active'
                        for i, r in enumerate(all_rewards):
                            writer.add_scalar(f"charts/episodic_return_{i}", r, global_step)

                        # these are original env returns
                        if "org_return" in info["episode"]:
                            writer.add_scalar("charts/episodic_env_return", info["episode"]["org_return"], global_step)
                        
                        # return of the actual chosen subpolicy
                        active_rewards = curr_episode_rewards[np.arange(args.max_env_steps), np.arange(args.num_envs), curr_episode_choices.squeeze()]
                        active_return = active_rewards.sum()
                        writer.add_scalar("charts/active_episodic_return", active_return, global_step)
                        for r_idx in range(n_rewards):
                            curr_reward = active_rewards[curr_episode_choices.squeeze() == r_idx]
                            curr_return = curr_reward.sum()
                            curr_len = len(curr_reward)
                            writer.add_scalar(f"charts/active_episodic_return_{r_idx}", curr_return, global_step)
                            writer.add_scalar(f"charts/activity_{r_idx}", curr_len/episode_step, global_step)



                        # TODO: add return for each subpolicy, where we only consider rewards when they were active
                    episode_step = 0
                    curr_episode_rewards = np.zeros((args.max_env_steps, args.num_envs, n_rewards))
                    # curr_episode_choices = np.zeros((args.max_env_steps, args.num_envs))
                    curr_episode_choices = np.ones((args.max_env_steps, args.num_envs), dtype=int) * -1

        elif episode_step >= args.max_env_steps:
            #TODO: reset env etc.
            raise ValueError("Episode step exceeded max env steps")
            episode_step = 0
            curr_episode_rewards = np.zeros((args.max_env_steps, args.num_envs, n_rewards))
            # curr_episode_choices = np.zeros((args.max_env_steps, args.num_envs))
            curr_episode_choices = np.ones((args.max_env_steps, args.num_envs)) * -1
                        

        if (global_step // args.num_envs) % args.save_model_steps == 0:
            save_path = f"{model_save_dir}/step_{global_step}.pt"
            save(save_path, q_networks, optimizers, global_step, args)
            print(f"Model saved at step {global_step} to {save_path}")


        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if (global_step // args.num_envs) % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                # data.rewards are now shape (-1, 1, n_rewards)
                # TODO: do computation for each subpolicy 
                for i, q_network in enumerate(q_networks):
                    next_obs = data.next_observations
                    dones = data.dones.flatten()
                    # choose subpolicy-specific reward
                    rewards = data.rewards[..., i].flatten()
                    # with torch.no_grad():
                    #     target_max, _ = target_network(data.next_observations).max(dim=1)
                    #     td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    # old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                    with torch.no_grad():
                        target_max, _ = target_network(next_obs.view(args.batch_size, -1)).max(dim=1)
                        td_target = rewards + args.gamma * target_max * (1 - dones)
                    old_val = q_network(data.observations.view(args.batch_size, -1)).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target.to(torch.float32), old_val)

                    if global_step % 100 == 0:
                        writer.add_scalar(f"losses/td_loss_{i}", loss, global_step)
                        writer.add_scalar(f"losses/q_values_{i}", old_val.mean().item(), global_step)

                    # optimize the model
                    optimizers[i].zero_grad()
                    loss.backward()
                    optimizers[i].step()

                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # update target network
            if (global_step // args.num_envs) % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    envs.close()
    writer.close()