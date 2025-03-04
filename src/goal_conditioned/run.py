import torch
import gymnasium as gym
from gc_ppo import Agent, Args
from logic_options.envs.common import make_hackatari_env
import importlib
import sys

def evaluate_agent(checkpoint_path, env_id, num_episodes=10, render=False):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    args = checkpoint['args']
    hackatari_args = checkpoint['hackatari_args']
    num_subpols = checkpoint['num_subpols']
    device = torch.device("cuda:15" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create the environment
    env = make_hackatari_env(env_id, 0, **hackatari_args)()

    # Initialize the agents
    agents = [Agent(env, args.norm_obs).to(device) for _ in range(num_subpols)]
    for i, agent in enumerate(agents):
        agent.load_state_dict(checkpoint['model_state_dict'][i])

    # Load meta-policy function or agent
    meta_policy = None
    meta_policy_func = None
    if args.neural_meta_policy:
        meta_policy = Agent(env, device, norm_obs=False, meta=True, num_subpolicies=num_subpols).to(device)
        meta_policy.load_state_dict(checkpoint['model_state_dict'][-1])
    else:
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
        last_option_choice = None
        while not done:
            if render:
                env.render()
            with torch.no_grad():
                if meta_policy is not None:
                    option_choices, _, _, _ = meta_policy.get_action_and_value(obs.view(1, -1))
                else:
                    option_choices = torch.tensor(meta_policy_func(obs)).to(device)
                
                if last_option_choice is None or last_option_choice.item() != option_choices.item():
                    print(f"Option choice changed to: {option_choices.item()}")
                    last_option_choice = option_choices

                action, _, _, _ = agents[option_choices.item()].get_action_and_value(obs.view(1, -1))
            obs, reward, done, _, _ = env.step(action.cpu().numpy())
            obs = torch.Tensor(obs).to(device)
            episode_reward += reward
        print(f"Episode {episode + 1}: Reward: {episode_reward}")

if __name__ == "__main__":
    checkpoint_path = "models/ALE/Seaquest-v5__gc_ppo__1__1741001110_step_10000.pt"
    env_id = "ALE/Seaquest-v5"
    evaluate_agent(checkpoint_path, env_id, render=True)
