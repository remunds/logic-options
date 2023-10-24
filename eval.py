from options.ppo import load_agent
from utils.callbacks import evaluate_policy


if __name__ == "__main__":
    name = "scobot-paper/unpruned-2"
    env_name = "ALE/Seaquest-v5"
    n_envs = 10
    n_eval_episodes = 100

    model = load_agent(name, env_name, n_envs=n_envs)
    env = model.get_env()

    print(f"Evaluating '{name}' on {env_name} for {n_eval_episodes} episodes... ", end="")

    mean_return, std_return = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)

    print(f"Done.")
    print(f"\tReturn: \t{mean_return:.2f} +/- {std_return:.2f}")
