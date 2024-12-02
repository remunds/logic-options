from logic_options.envs.common import init_vec_env
import numpy as np


ENV_NAME = "MeetingRoom"
ENV_ARGS = {
    "n_buildings": 1,
    "n_floors": 4,
    "max_steps": 100,
}
EARLY_STOP = None


if __name__ == "__main__":
    n_envs = 10
    n_eval_episodes = 1000

    env = init_vec_env(ENV_NAME, n_envs, seed=0, settings=ENV_ARGS)
    action_space = env.action_space

    print(f"Evaluating random policy on {ENV_NAME} for {n_eval_episodes} episodes... ", end="")

    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    last_positive_rewards = np.zeros(n_envs)
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    while (episode_counts < episode_count_targets).any():
        actions = np.asarray([action_space.sample() for _ in range(n_envs)])
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1

        last_positive_rewards[rewards > 0] = 0
        last_positive_rewards[rewards <= 0] += 1

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if EARLY_STOP is not None and last_positive_rewards[i] >= EARLY_STOP:
                    dones[i] = True
                    stop_early = True
                    new_observations[i] = env.env_method("reset", indices=[i])[0][0]
                else:
                    stop_early = False

                if dones[i]:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

    mean_return = float(np.mean(episode_rewards))
    std_return = float(np.std(episode_rewards))
    mean_lengths = float(np.mean(episode_lengths))
    std_lengths = float(np.std(episode_lengths))

    print(f"Done.")
    print(f"\tReturn: \t{mean_return:.2f} +/- {std_return:.2f}")
    print(f"\tLengths: \t{mean_lengths:.2f} +/- {std_lengths:.2f}")
