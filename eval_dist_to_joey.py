import numpy as np
import torch as th

from stable_baselines3.common.vec_env import unwrap_vec_normalize
from scobi.reward_shaping import _get_game_objects_by_category
from scobi.core import Environment as ScobiEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper

from envs.common import init_vec_env
from options.ppo import load_agent


def get_distance_to_joey(game_objects):
    player, child = _get_game_objects_by_category(game_objects, ["Player", "Child"])

    platform = np.ceil((player.xy[1] - player.h - 16) / 48)  # 0: topmost, 3: lowest platform

    if platform == 0:
        distance = 0
    else:
        if platform in [1, 3]:
            curr_platform_dist = abs(player.xy[0] - 132)
        else:
            curr_platform_dist = abs(player.xy[0] - 20)
        distance = abs(child.xy[1] - player.xy[1]) + max(platform - 1, 0) * 112 + curr_platform_dist

    return distance


if __name__ == "__main__":
    names = ["dist-to-joey/neural-0",
             "dist-to-joey/neural-1",
             "dist-to-joey/neural-2"]
    env_name = "ALE/Kangaroo-v5"
    n_envs = 1
    n_eval_episodes = 1
    deterministic = True
    seed = 25

    episode_best_distances = []

    for name in names:
        model = load_agent(name, env_name, n_envs=1)
        env_orig = model.get_env()
        raw_env = env_orig.envs[0].env
        is_object_centric = isinstance(raw_env, ScobiEnv)

        # The environment to operate on during eval
        if is_object_centric:
            env = env_orig
            vec_norm = None
            raw_ocatari_env = raw_env.oc_env
        else:
            env = init_vec_env(env_name, n_envs=n_envs, seed=seed, object_centric=True, no_scobi=True)
            vec_norm = unwrap_vec_normalize(env_orig)
            raw_ocatari_env = env.envs[0].env

        # model_dir = to_model_dir(name, env_name)
        # vec_norm_path = model_dir / "checkpoints" / "best_vecnormalize.pkl"
        # if os.path.exists(vec_norm_path):
        #     env = VecNormalize.load(vec_norm_path, venv=env)

        print(f"Evaluating '{name}' on {env_name} for {n_eval_episodes} episodes... ", end="")

        # Get mean and standard deviation of distance to Joey
        episode_rewards = []
        episode_lengths = []

        episode_counts = np.zeros(n_envs, dtype="int")
        # Divides episodes among different sub environments in the vector as evenly as possible
        episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

        current_rewards = np.zeros(n_envs)
        current_best_distances = np.zeros(n_envs) + 1000
        current_lengths = np.zeros(n_envs, dtype="int")
        observations = env.reset()
        if vec_norm is not None:
            observations = vec_norm.normalize_obs(observations)
        states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        option_terminations = th.ones(env.num_envs, model.hierarchy_size, device=model.device, dtype=th.bool)
        options = th.zeros(env.num_envs, model.hierarchy_size, device=model.device, dtype=th.long)

        while (episode_counts < episode_count_targets).any():
            (options, actions), _, _ = model.forward_all(observations, options, option_terminations, deterministic)

            new_observations, rewards, dones, infos = env.step(actions)
            if vec_norm is not None:
                new_observations = vec_norm.normalize_obs(new_observations)
            current_rewards += rewards
            current_lengths += 1

            for i in range(n_envs):
                game_objects = raw_ocatari_env.objects
                d = get_distance_to_joey(game_objects)
                current_best_distances[i] = min(current_best_distances[i], d)

                if episode_counts[i] < episode_count_targets[i]:
                    # unpack values so that the callback can access the local variables
                    reward = rewards[i]
                    done = dones[i]
                    info = infos[i]
                    episode_starts[i] = done

                    if dones[i]:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_best_distances.append(current_best_distances[i])
                        episode_counts[i] += 1
                        current_rewards[i] = 0
                        current_lengths[i] = 0
                        current_best_distances[i] = 1000

            option_terminations, _ = model.forward_all_terminators(new_observations, options)
            option_terminations[dones] = True

            observations = new_observations

        best_completed_distances = (470 - np.array(episode_best_distances)) / 470

    mean = float(np.mean(best_completed_distances))
    std = float(np.std(best_completed_distances))

    print(f"Done.")
    print(f"\tBest completed distance to Joey: \t{mean:.2f} +/- {std:.2f} (1.00 means 100%, i.e., Joey reached)")
