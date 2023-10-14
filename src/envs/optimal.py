"""Computes the average of the shortest path for MeetingRoom."""
import numpy as np

from meeting_room import MeetingRoom

N_SAMPLES = 1000

env = MeetingRoom(
    n_buildings=4,
    n_floors=4,
)

distances = np.zeros(N_SAMPLES, dtype=int)

for n in range(N_SAMPLES):
    env.reset()
    distances[n] = env.distance_map[tuple(env.current_pos)]

print(f"Mean distance {np.average(distances)} +/- {np.std(distances)}")
