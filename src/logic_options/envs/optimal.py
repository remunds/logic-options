"""Computes the average of the shortest path for MeetingRoom."""
import numpy as np

from logic_options.envs.meeting_room import MeetingRoom

N_SAMPLES = 1000

env = MeetingRoom(
    n_buildings=1,
    n_floors=4,
)

distances = np.zeros(N_SAMPLES, dtype=int)
returns = np.zeros(N_SAMPLES, dtype=float)

for n in range(N_SAMPLES):
    env.reset()
    distance = env.distance_map[tuple(env.current_pos)]
    floor_diff = abs(env.current_pos[1] - env.target[1])
    ret = distance / 100 + floor_diff * 0.2 + 1  # add building switch bonus
    distances[n] = distance
    returns[n] = ret


print(f"Mean distance {np.average(distances)} +/- {np.std(distances)}")
print(f"Mean return {np.average(returns)} +/- {np.std(returns)}")
