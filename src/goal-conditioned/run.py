import torch
import gymnasium as gym
from gc_ppo import Agent, Args
from logic_options.envs.common import make_hackatari_env
import importlib
import sys

def evaluate_agent(checkpoint_path, env_id, num_episodes=10):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    args = checkpoint['args']
    device = torch.device("cuda:15" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create the environment
    hackatari_args = {
        # "rewardfunc_path": ["in/reward_funcs/seaquest/hud/fight_enemies.py",
        #                     "in/reward_funcs/seaquest/hud/collect_divers.py",
        #                     "in/reward_funcs/seaquest/hud/surface.py",
        #                     ],
        # "rewardfunc_path": "in/reward_funcs/seaquest/hud/hackatari_reward.py"
    }
    env = make_hackatari_env(env_id, 0, **hackatari_args)()

    # Initialize the agents
    num_subpols = 3
    agents = [Agent(env, args.norm_obs).to(device) for _ in range(num_subpols)]
    for i, agent in enumerate(agents):
        agent.load_state_dict(checkpoint['model_state_dict'][i])


    # Load meta-policy function
    meta_function_path = "in/logic/llm/seaquest-meta-policy.py"
    spec = importlib.util.spec_from_file_location("meta_policy", meta_function_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["meta_policy"] = module
    spec.loader.exec_module(module)
    meta_policy_func = module.meta_policy

    # Evaluate the agent
    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.Tensor(obs).to(device)
        done = False
        episode_reward = 0
        while not done:
            with torch.no_grad():
                option_choices = torch.tensor(meta_policy_func(obs)).to(device)
                action, _, _, _ = agents[option_choices.item()].get_action_and_value(obs)
            obs, reward, done, _, _ = env.step(action.cpu().numpy())
            obs = torch.Tensor(obs).to(device)
            episode_reward += reward
        print(f"Episode {episode + 1}: Reward: {episode_reward}")

if __name__ == "__main__":
    checkpoint_path = "models/ALE/Seaquest-v5__gc_ppo__1__1741001110_step_10000.pt"
    env_id = "ALE/Seaquest-v5"
    evaluate_agent(checkpoint_path, env_id)
