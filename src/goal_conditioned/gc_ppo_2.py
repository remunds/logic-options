# CleanRL - PPO
# Idea: optimize all subpolicies even if another one was active 
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from logic_options.envs.common import make_hackatari_env
from logic_options.utils.normalize_obs_torch import RunningMeanStd
from rtpt import RTPT

import importlib
import sys
import yaml

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "ALE/Seaquest-v5"
    """the id of the environment"""
    total_timesteps: int = 20_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    norm_obs: bool = True
    """Toggles observation normalization, my implementation"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    save_model_steps: int = 1_000_000
    """number of steps between saving the model"""
    checkpoint_path: str = None #"models/ALE/Seaquest-v5__gc_ppo__1__1741001110_step_10000.pt" 
    """path to the checkpoint file to resume training from"""
    neural_meta_policy: bool = False
    """if toggled, a neural meta policy will be used, otherwise a function will be used"""
    # meta_policy_path: str = None
    meta_policy_path = "in/logic/llm/seaquest-meta-policy.py"
    """path to the meta policy function"""
    num_subpolicies: int = 3
    """number of subpolicies"""
    # rewardfunc_path: str = None 
    rewardfunc_path = ["in/reward_funcs/seaquest/hud/fight_enemies.py",
                        "in/reward_funcs/seaquest/hud/collect_divers.py",
                        "in/reward_funcs/seaquest/hud/surface.py",
                       ]
    # rewardfunc_path = "in/reward_funcs/seaquest/hud/hackatari_reward.py"
    """path to the reward function(s)"""
    args_file: str = None #"runs/ALE/Seaquest-v5__gc_ppo__1__1741019777/args.yaml"#None
    """path to the args file to load arguments from"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, device, norm_obs=False, meta=False, num_subpolicies=None):
        super().__init__()
        self.meta = meta
        if meta and num_subpolicies is None:
            raise ValueError("num_subpolicies must be specified if meta=True")
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_subpolicies if meta else envs.single_action_space.n), std=0.01),
        )
        self.norm_obs = norm_obs
        if norm_obs:
            shape = (np.prod(envs.single_observation_space.shape),)
            self.obs_rms = RunningMeanStd(shape=shape, device=device)
            self.epsilon = 1e-8

    def _rms_normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        # only update if batch size is > 1
        # prev_shape = obs.shape
        # flatten all but the first dimension
        # prev_obs = obs.view(prev_shape[0], -1).to(torch.float32)
        with torch.no_grad():
            if obs.shape[0] > 1:
                self.obs_rms.update(obs)
            new_obs = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + self.epsilon)
        return new_obs.to(torch.float32)#.view(prev_shape)


    def get_value(self, x):
        if self.norm_obs:
            x = self._rms_normalize(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if self.norm_obs:
            x = self._rms_normalize(x)
        # check if any nan
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def save(save_path, agents, optimizers, iteration, args, hackatari_args, num_subpols):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
    torch.save({
        'iteration': iteration,
        'model_state_dict': [agent.state_dict() for agent in agents],
        'optimizer_state_dict': [optimizer.state_dict() for optimizer in optimizers],
        'args': args,
        'hackatari_args': hackatari_args,
        'num_subpols': num_subpols,
    }, save_path)

def load_gc_agent(env_name, run_id, device, best_model=True):
    model_name = "best_return"
    if not best_model:
        # find latest model(highest number)
        model_name = max([f for f in os.listdir(f"runs/{env_name}__{run_id}/models") if f.endswith(".pt")])
        # remove .pt
        model_name = model_name[:-3]

    save_path = f"runs/{env_name}__{run_id}/models/best_model.pt"
    checkpoint = torch.load(save_path)
    args = checkpoint['args']
    hackatari_args = checkpoint['hackatari_args']
    num_subpols = checkpoint['num_subpols']

    # Create the environment
    env = make_hackatari_env(args.env_id, 0, **hackatari_args)()

    # Initialize the agents
    agents = [Agent(env, args.norm_obs).to(device) for _ in range(num_subpols)]
    for i, agent in enumerate(agents):
        agent.load_state_dict(checkpoint['model_state_dict'][i])

    return agents, env

def save_args(args, save_path):
    with open(save_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

def load_args(args_file):
    with open(args_file, 'r') as f:
        args_dict = yaml.safe_load(f)
    return Args(**args_dict)

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.args_file:
        print(f"Loading arguments from {args.args_file}")
        args = load_args(args.args_file)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    device = torch.device("cuda:15")
    hackatari_args = {
        "rewardfunc_path": args.rewardfunc_path 
    }
    n_rewards = len(hackatari_args["rewardfunc_path"]) if hackatari_args["rewardfunc_path"] is not None else 0

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_hackatari_env(args.env_id, i, **hackatari_args) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agents = [Agent(envs, device, args.norm_obs).to(device) for _ in range(args.num_subpolicies)]

    # neural meta policy
    if args.neural_meta_policy:
        meta_policy = Agent(envs, device, norm_obs=False, meta=True, num_subpolicies=3).to(device)
        agents.append(meta_policy)
        meta_policy_func = None

    # load meta-policy function
    elif args.meta_policy_path is not None:
        spec = importlib.util.spec_from_file_location("meta_policy", args.meta_policy_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["meta_policy"] = module
        spec.loader.exec_module(module)
        meta_policy_func = module.meta_policy
        meta_policy = None
    
    else:
        raise ValueError("Either neural_meta_policy or meta_policy_path must be specified")

    optimizers = [optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5) for agent in agents]

    # Load checkpoint if provided
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path)
        for i, agent in enumerate(agents):
            agent.load_state_dict(checkpoint['model_state_dict'][i])
        for i, optimizer in enumerate(optimizers):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'][i])
        iteration = checkpoint['iteration']
        print(f"Resumed training from checkpoint {args.checkpoint_path} at iteration {iteration}")
    else:
        iteration = 1

    highest_episodic_return = float('-inf')

    # ALGO Logic: Storage setup
    flat_obs_shape = np.array(envs.single_observation_space.shape).prod().item()
    obs = torch.zeros((args.num_steps, args.num_envs, flat_obs_shape)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs, args.num_subpolicies)).to(device)
    meta_logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs, args.num_subpolicies)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs, args.num_subpolicies)).to(device)
    meta_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    subpolicies = torch.zeros((args.num_steps, args.num_envs)).to(device)

    subpolicy_activity = torch.zeros((args.num_iterations, args.num_subpolicies)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    rtpt = RTPT(name_initials='RE', experiment_name='goal_cond_ppo', max_iterations=args.num_iterations)
    rtpt.start()

    for iteration in range(iteration, args.num_iterations + 1):
        rtpt.step()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            for optimizer in optimizers:
                optimizer.param_groups[0]["lr"] = lrnow

        # collect trajectories
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs.view(args.num_envs, -1)
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if meta_policy is not None:
                    option_choices, _ , _, _= meta_policy.get_action_and_value(obs[step])
                elif meta_policy_func is not None:
                    option_choices = torch.tensor(meta_policy_func(next_obs)).to(device)
                else:
                    raise ValueError("meta_policy or meta_policy_func must be specified")

                subpolicies[step] = option_choices
                # for action, we only need the one that is active
                # rest: for all subpolicies
                action = torch.zeros(args.num_envs).to(device, dtype=torch.long)
                logprob = torch.zeros(args.num_envs, args.num_subpolicies).to(device)
                meta_logprob = torch.zeros(args.num_envs).to(device)
                value = torch.zeros((args.num_envs, args.num_subpolicies)).to(device)
                meta_value = torch.zeros(args.num_envs).to(device)

                # here we find the actual action that is taken
                for i, agent in enumerate(agents):
                    # for meta-policy, we only collect logprob for building the ratio during optimization 
                    if agent.meta:
                        _, meta_logprob_l, _, meta_value_l = agent.get_action_and_value(obs[step])
                        meta_logprob = meta_logprob_l.to(torch.float32)
                        meta_value = meta_value_l.to(torch.float32).view(-1)
                        continue

                    # Note: could filter here 
                    action_l, _ , _, value_l = agent.get_action_and_value(obs[step])
                    # set action only if current subpolicy is the active one
                    env_idxs = torch.where(option_choices == i)[0]

                    action[env_idxs] = action_l[env_idxs]
                    # logprobs are computed below
                    # always store value from critic (all subpolicies)
                    value[:, i] = value_l.to(torch.float32).view(-1)

                # here we compute the logprobs of the actual taken actions 
                for i, agent in enumerate(agents):
                    if agent.meta:
                        continue
                    _, logprob_actual, _, _ = agent.get_action_and_value(obs[step], action)
                    logprob[:, i] = logprob_actual.to(torch.float32)
                
                # get actions according to the correct subpolicy 
                values[step] = value #(args.num_envs, args.num_subpolicies)
                actions[step] = action.to(torch.float32) # (args.num_envs)
                logprobs[step] = logprob # (args.num_envs, args.num_subpolicies)
                meta_logprobs[step] = meta_logprob # (args.num_envs)
                meta_values[step] = meta_value # (args.num_envs)

            subpolicy_activity[iteration - 1] += torch.bincount(subpolicies[step].to(int), minlength=args.num_subpolicies)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            if "all_rewards" in infos:
                all_rewards = np.array([np.array(object=a_r) if a_r is not None else np.array([0.0 for _ in range(n_rewards)]) for a_r in infos["all_rewards"]])
                # (n_envs, n_rewards)
                reward = all_rewards
            else:
                raise ValueError("all_rewards not in infos")
            rewards[step] = torch.tensor(reward).to(device)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episodic_return = info['episode']['r']
                        print(f"global_step={global_step}, episodic_return={episodic_return}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        if episodic_return > highest_episodic_return:
                            highest_episodic_return = episodic_return
                            save_path = f"{model_save_dir}/best_return.pt"
                            save(save_path, agents, optimizers, iteration, args, hackatari_args, args.num_subpolicies)
                            print(f"New highest episodic return: {episodic_return}. Model saved to {save_path}")
                        if "all_rewards" in info["episode"] and isinstance(info["episode"]["all_rewards"], list):
                            all_rewards = info["episode"]["all_rewards"]
                            for i, r in enumerate(all_rewards):
                                writer.add_scalar(f"charts/episodic_return_{i}", r, global_step)

            if global_step % args.save_model_steps == 0:
                save_path = f"{model_save_dir}/step_{global_step}.pt"
                save(save_path, agents, optimizers, iteration, args, hackatari_args, args.num_subpolicies)
                print(f"Model saved at step {global_step} to {save_path}")

        subpolicy_activity[iteration - 1] /= args.num_steps * args.num_envs

        for i in range(args.num_subpolicies):
            writer.add_scalar(f"charts/activity_{i}", subpolicy_activity[iteration - 1, i].item(), global_step)

        # bootstrap value if not done
        # Compute GAE
        for i, agent in enumerate(agents):
            with torch.no_grad():
                # meta: select rewards of actual selected subpolicy
                # Alternatively, could use seperate reward for meta-policy (e.g. game reward)
                if agent.meta:
                    agent_rewards = torch.gather(rewards, dim=-1, index=subpolicies.long().unsqueeze(-1)).squeeze(-1)  # Shape (128, 8)
                    agent_values = meta_values
                else:
                    agent_rewards = rewards[..., i]
                    agent_values = values[..., i]
                next_value = agent.get_value(next_obs.view(args.num_envs, -1)).reshape(1, -1)
                advantages = torch.zeros_like(agent_rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = agent_values[t + 1]
                    delta = agent_rewards[t] + args.gamma * nextvalues * nextnonterminal - agent_values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + agent_values

            if not agent.meta:
                a_logprobs = logprobs[..., i]
                b_logprobs = a_logprobs.reshape(-1)
                a_values = values[..., i]
                b_values = a_values.reshape(-1)
            else:
                b_values = meta_values.reshape(-1)

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_meta_logprobs = meta_logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            if agent.meta:
                b_subpolicies = subpolicies.reshape(-1)

            batch_size = b_obs.shape[0]
            minibatch_size = batch_size // args.num_minibatches
            if batch_size == 0 or minibatch_size == 0:
                print(f"Skipping subpolicy {i}, batch_size={batch_size}, minibatch_size={minibatch_size}") 
                continue

            # Optimizing the policy and value network
            b_inds = np.arange(batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]
                    mb_obs = b_obs[mb_inds].view(b_obs[mb_inds].shape[0], -1)
                    if agent.meta:
                        # actions for meta are subpolicies
                        mb_actions = b_subpolicies[mb_inds].long()
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
                        logratio = newlogprob - b_meta_logprobs[mb_inds]
                    else:
                        # Should we compute the ratio with the actually used action
                        # or with the action that the subpolicy would have taken?
                        # Note that the ratio is later multiplied with the advantage
                        # which is computed with the actual action
                        # So for now: compute ratio with actual action
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, b_actions.long()[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        std = 1 if mb_advantages.shape[0] == 1 else mb_advantages.std()
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (std + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    if torch.isnan(pg_loss):
                        print("Nan in pg_loss")
                        continue

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    # print(i, loss.item(), pg_loss.item(), entropy_loss.item(), v_loss.item(), approx_kl.item(), old_approx_kl.item())

                    optimizers[i].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizers[i].step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizers[0].param_groups[0]["lr"], global_step)
            writer.add_scalar(f"losses/value_loss_{i}", v_loss.item(), global_step)
            writer.add_scalar(f"losses/policy_loss_{i}", pg_loss.item(), global_step)
            writer.add_scalar(f"losses/entropy_{i}", entropy_loss.item(), global_step)
            writer.add_scalar(f"losses/old_approx_kl_{i}", old_approx_kl.item(), global_step)
            writer.add_scalar(f"losses/approx_kl_{i}", approx_kl.item(), global_step)
            writer.add_scalar(f"losses/clipfrac_{i}", np.mean(clipfracs), global_step)
            writer.add_scalar(f"losses/explained_variance_{i}", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()