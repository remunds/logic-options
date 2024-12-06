from typing import SupportsFloat, Any, Dict, Tuple, Sequence
from enum import Enum
from queue import Queue

import pygame
import numpy as np
from gymnasium import Env, spaces
from gymnasium.core import ActType, ObsType
from stable_baselines3.common.vec_env import VecNormalize
import torch as th

from logic_options.options.option import Option
from logic_options.utils.render import draw_arrow, draw_label

PLAYER_VIEW_SIZE = 7  # odd number

MARGIN = 50
FIELD_SIZE = 40
FLOOR_HEIGHT = 25
BUILDING_WIDTH = 50
BUILDING_MARGIN = 40
BORDER_WIDTH = 1

PLAYER_COLOR = "#0083CC"
TARGET_COLOR = "#FDCA00"
ENTRANCE_COLOR = "#009D81"
ELEVATOR_COLOR = ENTRANCE_COLOR


class Action(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    NEXT = 4  # next level or building
    PREV = 5  # previous level or building


direction = {
    "north": np.array([0, -1]),
    "east": np.array([1, 0]),
    "south": np.array([0, 1]),
    "west": np.array([-1, 0]),
}


class MeetingRoom(Env):
    """Find the meeting room! It is located on a specific floor inside a specific building.

    :param n_buildings:
    :param n_floors:
    :param floor_shape: (width, height)
    :param render_mode:
    :param walls_fixed: if True, wall positions/floor maps remain the same over different
        episodes AND the walls are the same on all floors of a building
    """

    game_name = "meetingroom"
    action_space = spaces.Discrete(len(Action))
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    current_pos: np.array
    previous_pos: np.array
    target: np.array

    def __init__(self,
                 n_buildings: int = 4,
                 n_floors: int = 4,
                 floor_shape: [int, int] = None,
                 goal: str = "target",
                 max_steps: int = 100,
                 render_mode: str = None,
                 walls_fixed: bool = False):
        if goal not in ["target", "elevator", "entrance"]:
            raise ValueError(f"Unrecognized goal '{goal}'.")

        if floor_shape is None:
            floor_shape = np.array([11, 11], dtype=int)
        else:
            floor_shape = np.asarray(floor_shape)

        self.goal = goal
        self.render_mode = render_mode
        self.n_buildings = n_buildings
        self.n_floors = n_floors
        assert np.all(floor_shape[0] >= 5)
        self.floor_shape = floor_shape
        self.walls_fixed = walls_fixed

        obs_len = 9 if self.walls_fixed else 9 + PLAYER_VIEW_SIZE ** 2
        self.observation_space = spaces.Box(0, np.max(floor_shape), shape=(obs_len,), dtype=np.int32)

        self.map = np.zeros((self.n_buildings, self.n_floors, *floor_shape), dtype=int)
        self.elevators = np.zeros((self.n_buildings, 2), dtype=int)
        self.entrances = np.zeros((self.n_buildings, 2), dtype=int)
        self.distance_map = np.zeros((self.n_buildings, self.n_floors, *floor_shape), dtype=int) - 1

        self.map_generated = False
        self.goal_reached = False
        self.goal_state_showed = False
        self.terminated = False
        self.n_steps = 0
        self.max_steps = max_steps

        self._current_distance_to_target = -1
        self._previous_distance_to_target = -1
        self._previous_best_distance_to_target = -1

        # Option stats rendering
        self.render_option_history = False
        self.option = None
        self.vec_norm = None
        self.action = None
        self.policy = None
        self.option_pos = None
        self.option_history_map = np.zeros((self.n_buildings, self.n_floors, *floor_shape), dtype=int) - 1

        # Setup rendering
        self.window_shape = self.floor_shape * (FIELD_SIZE + BORDER_WIDTH) + 2 * BORDER_WIDTH + 2 * MARGIN
        self.window_shape[1] += self.n_floors * FLOOR_HEIGHT + MARGIN
        if self.render_mode == "human":
            self._init_pygame()
            self.window = pygame.display.set_mode(self.window_shape)
        else:
            self.window = pygame.Surface(self.window_shape)

    def _generate_map(self):
        """Renews map, entrances and elevators."""
        for b in range(self.n_buildings):
            elevator = self.np_random.integers(1, self.floor_shape - 1)
            entrance = self._generate_entrance_position()
            self.elevators[b] = elevator
            self.entrances[b] = entrance

            floor_base_map = self._generate_floor_map()

            for f in range(self.n_floors):
                if self.walls_fixed:
                    floor_map = floor_base_map.copy()
                else:
                    floor_map = self._generate_floor_map()

                # Remove walls in special cases
                if f == 0:
                    entrance_x, entrance_y = entrance
                    floor_map[entrance_x, entrance_y] = 0
                    c_dir = self._get_center_direction(*entrance)
                    floor_map[entrance_x + c_dir[0], entrance_y + c_dir[1]] = 0

                elevator_x, elevator_y = elevator
                floor_map[elevator_x, elevator_y] = 0
                if self._surrounded_by_walls(*elevator, floor_map):
                    c_dir = self._get_center_direction(*elevator)
                    floor_map[elevator_x + c_dir[0], elevator_y + c_dir[1]] = 0

                self.map[b, f] = floor_map

    def _generate_floor_map(self):
        """Creates an array encoding the floor's walls. Walls are represented
        by 1, empty space by 0."""
        floor_map = np.zeros(self.floor_shape)

        # Surrounding walls
        floor_map[[0, -1]] = 1
        floor_map[:, [0, -1]] = 1

        # Vertical wall
        wall_v = self.np_random.integers(2, self.floor_shape[1] - 2)
        floor_map[wall_v] = 1

        # Horizontal wall
        wall_h = self.np_random.integers(2, self.floor_shape[0] - 2, size=2)
        floor_map[:wall_v, wall_h[0]] = 1
        floor_map[wall_v:, wall_h[1]] = 1

        # Doorways
        # north
        n_choices = list(range(1, np.min(wall_h)))
        door_pos = self.np_random.choice(n_choices)
        floor_map[wall_v, door_pos] = 0

        # east
        n_choices = list(range(wall_v + 1, self.floor_shape[0] - 1))
        door_pos = self.np_random.choice(n_choices)
        floor_map[door_pos, wall_h[1]] = 0

        # south
        n_choices = list(range(np.max(wall_h) + 1, self.floor_shape[1] - 1))
        door_pos = self.np_random.choice(n_choices)
        floor_map[wall_v, door_pos] = 0

        # west
        n_choices = list(range(1, wall_v))
        door_pos = self.np_random.choice(n_choices)
        floor_map[door_pos, wall_h[0]] = 0

        return floor_map

    def _generate_entrance_position(self):
        # Pick one of the four sides of the building (0 north, 1 east etc.)
        side = self.np_random.integers(0, 4)

        # Determine entrance position inside wall
        wall_length = self.floor_shape[0] if side % 2 else self.floor_shape[1]
        pos_in_wall = self.np_random.integers(1, wall_length - 1)

        # Return final entrance door position
        if side == 0:
            x, y = 0, pos_in_wall
        elif side == 1:
            x, y = pos_in_wall, self.floor_shape[1] - 1
        elif side == 2:
            x, y = self.floor_shape[0] - 1, pos_in_wall
        else:
            x, y = pos_in_wall, 0
        return np.array([x, y], dtype=int)

    def reset(
            self,
            *,
            seed: int = None,
            options: Dict[str, Any] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        super().reset(seed=seed)

        if not self.map_generated or not self.walls_fixed:
            self._generate_map()
            self.map_generated = True
        self.current_pos = self._pick_random_position()
        self.previous_pos = None

        target = self._pick_random_position()
        while np.all(target == self.current_pos):
            target = self._pick_random_position()
        self.target = target
        self._compute_distances()
        self._best_distance_to_target = np.inf
        self.terminated = False
        self.goal_reached = False
        self.goal_state_showed = False
        self.n_steps = 0
        self.option_history_map[:] = -1
        if self.render_mode == "human":
            self.render()
        return self._get_observation(), self._get_info()

    def _pick_random_position(self):
        b = self.np_random.integers(self.n_buildings)
        f = self.np_random.integers(self.n_floors)

        # Find valid position on floor, not occupied by a wall
        while True:
            x, y = self.np_random.integers(1, self.floor_shape - 2, size=2)
            if not self.map[b, f, x, y]:  # empty spot
                break

        return np.array([b, f, x, y])

    def _get_observation(self):
        """CAUTION: If you change this function, also update
        logic.valuation.meeting_room.py accordingly."""
        obs = [
            self.target - self.current_pos,
            self._get_current_elevator() - self.current_pos[2:4],
            self._get_current_entrance() - self.current_pos[2:4],
            self.current_pos[1],
        ]
        if not self.walls_fixed:
            obs.append(self._get_player_view().flatten())
        return np.hstack(obs)

    def _get_player_view(self):
        b, f, x, y = self.current_pos
        view_distance = PLAYER_VIEW_SIZE // 2
        view = self.map[b, f,
                   max(x - view_distance, 0):x + view_distance + 1,
                   max(y - view_distance, 0):y + view_distance + 1
               ]
        padded_view = np.ones((PLAYER_VIEW_SIZE, PLAYER_VIEW_SIZE))
        x_p = max(view_distance - x, 0)
        y_p = max(view_distance - y, 0)
        padded_view[x_p: x_p + view.shape[0], y_p: y_p + view.shape[1]] = view
        return padded_view

    def _get_info(self):
        return dict()

    def _update_best_distance(self):
        new_distance = self.distance_map[tuple(self.current_pos)]
        if new_distance < self._best_distance_to_target:
            self._best_distance_to_target = new_distance
            return True
        else:
            return False

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        self._act(Action(action))
        new_improvement = self._update_best_distance()
        goal_reached = self._check_goal_reached()
        self.goal_reached |= goal_reached
        obs = self._get_observation()
        reward = self._get_reward(goal_reached=goal_reached, new_improvement=new_improvement)
        self.n_steps += 1
        truncated = self.n_steps >= self.max_steps
        self.terminated |= self.goal_reached | truncated
        if self.render_mode in ["human", "rgb_array"] and self.goal_reached and not self.goal_state_showed:
            # Allow the human viewer to see that agent reached the goal
            self.terminated = False
            self.goal_state_showed = True
        info = self._get_info()
        return obs, reward, self.terminated, truncated, info

    def _check_goal_reached(self) -> bool:
        if not self.goal_reached:
            if self.goal == "target":
                return np.all(self.current_pos == self.target)
            elif self.goal == "elevator":
                return np.all(self.current_pos[2:] == self._get_current_elevator())
            else:  # entrance
                return np.all(self.current_pos[2:] == self._get_current_entrance())
        else:
            return False

    def _compute_distances(self):
        """Fills the distance map with the respective distance values
        to the target."""
        self.distance_map[:] = -1

        if self.goal == "target":
            goal = self.target
        elif self.goal == "elevator":
            goal = np.array([*self.current_pos[:2], *self._get_current_elevator()])
        else:  # entrance
            goal = np.array([self.current_pos[0], 0, *self._get_current_entrance()])
        self.distance_map[tuple(goal)] = 0

        # Dijkstra shortest path computation
        queue = Queue()
        queue.put(goal)
        while not queue.empty():
            position = queue.get()
            d = self.distance_map[tuple(position)]
            neighbors = self._get_neighbors(position)
            for neighbor in neighbors:
                if self.distance_map[tuple(neighbor)] == -1:
                    self.distance_map[tuple(neighbor)] = d + 1
                    queue.put(neighbor)

    def _get_reward(self, goal_reached: bool, new_improvement: bool):
        """Compares previous position with current position and gives a reward
        according to the change of the distance to the target. Penalizes non-movement."""
        # Distance reward
        previous_distance = self.distance_map[tuple(self.previous_pos)]
        current_distance = self.distance_map[tuple(self.current_pos)]
        distance_diff = previous_distance - current_distance
        # if distance_diff > 0:
        #     distance_diff = 0
        distance_reward = distance_diff / 100

        improvement_reward = 1 if new_improvement else 0

        # Floor switch reward
        if self.goal == "target":
            if self.target[0] == self.current_pos[0]:  # player in target building
                floor_diff = abs(self.target[1] - self.previous_pos[1]) - abs(self.target[1] - self.current_pos[1])
            else:
                floor_diff = self.previous_pos[1] - self.current_pos[1]
            floor_switch_bonus = floor_diff * 0.2
        else:
            floor_switch_bonus = 0

        # Building switch reward
        if self.goal == "target":
            building_diff = abs(self.target[0] - self.previous_pos[0]) - abs(self.target[0] - self.current_pos[0])
            building_switch_bonus = building_diff * 0.5
        else:
            building_switch_bonus = 0

        # Target reward
        target_reward = 1 if goal_reached else 0

        return target_reward + floor_switch_bonus + building_switch_bonus + distance_reward

    def _at_elevator(self, position):
        b, f, x, y = position
        elevator = self.elevators[b]
        return np.all(elevator == [x, y])

    def _at_entrance(self, position):
        b, f, x, y = position
        entrance = self.entrances[b]
        return f == 0 and np.all(entrance == [x, y])

    def _get_neighbors(self, position):
        """Returns the coordinates of all non-wall neighbor fields reachable from position."""
        b, f, x, y = position

        # Gather possible neighbor candidates
        candidates = []

        for d in direction.values():  # simple step
            candidate = position.copy()
            candidate[2:4] += d
            if np.all(0 <= candidate) and np.all(candidate[2:4] < self.floor_shape):
                candidates.append(candidate)

        if self._at_elevator(position):  # floor switch
            for v in [-1, 1]:
                next_f = f + v
                if 0 <= next_f < self.n_floors:
                    candidates.append(np.array([b, next_f, x, y]))

        elif self._at_entrance(position):  # building switch
            for v in [-1, 1]:
                next_b = b + v
                if 0 <= next_b < self.n_buildings:
                    next_entrance = self.entrances[next_b]
                    candidates.append(np.array([next_b, 0, *next_entrance]))

        # Remove neighbors with walls
        neighbors = []
        for candidate in candidates:
            if not self.map[tuple(candidate)]:
                neighbors.append(candidate)

        return neighbors

    def _act(self, action: Action):
        self.previous_pos = self.current_pos.copy()
        b, f, x, y = self.current_pos
        elevator = self._get_current_elevator()
        entrance = self._get_current_entrance()

        if action in [Action.NORTH, Action.EAST, Action.SOUTH, Action.WEST]:
            next_pos = None
            if action == Action.NORTH:
                next_pos = self.current_pos[2:4] + direction["north"]
            elif action == Action.EAST:
                next_pos = self.current_pos[2:4] + direction["east"]
            elif action == Action.SOUTH:
                next_pos = self.current_pos[2:4] + direction["south"]
            elif action == Action.WEST:
                next_pos = self.current_pos[2:4] + direction["west"]

            if self._is_valid_next_step(*next_pos):
                self.current_pos[2:4] = next_pos

        elif action in [Action.NEXT, Action.PREV]:
            if np.all(self.current_pos[2:4] == elevator):
                if action == Action.NEXT and f < self.n_floors - 1:
                    self.current_pos[1] += 1
                elif action == Action.PREV and f > 0:
                    self.current_pos[1] -= 1

            elif np.all(self.current_pos[2:4] == entrance) and f == 0:
                if action == Action.NEXT and b < self.n_buildings - 1:
                    self.current_pos[0] += 1
                elif action == Action.PREV and b > 0:
                    self.current_pos[0] -= 1
                entrance = self._get_current_entrance()
                self.current_pos[2:4] = entrance

    def _is_valid_next_step(self, x, y):
        b, f, _, _ = self.current_pos
        return 0 <= x < self.floor_shape[0] and \
            0 <= y < self.floor_shape[1] and not self.map[b, f, x, y]

    def _get_current_entrance(self):
        b = self.current_pos[0]
        return self.entrances[b]

    def _get_current_elevator(self):
        b = self.current_pos[0]
        return self.elevators[b]

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Meeting Room")
        self.clock = pygame.time.Clock()

    def render(self):
        self._render_frame()
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.window).transpose((1, 0, 2))
        elif self.previous_pos is None or not np.all(self.current_pos == self.previous_pos):
            self._display_frame()
            self.clock.tick(self.metadata["video.frames_per_second"])

    def _render_frame(self):
        self.window.fill((255, 255, 255))  # clear the entire window
        self._render_grid()
        self._render_buildings()

    def _render_grid(self):
        b, f, _, _ = self.current_pos
        height, width = self.floor_shape

        # Draw grid background
        grid_width = width * (FIELD_SIZE + BORDER_WIDTH) + BORDER_WIDTH
        grid_height = height * (FIELD_SIZE + BORDER_WIDTH) + BORDER_WIDTH
        pygame.draw.rect(self.window, (0, 0, 0), [MARGIN, MARGIN, grid_width, grid_height])

        entrance = self._get_current_entrance()
        elevator = self._get_current_elevator()

        # Draw all fields
        for x in range(width):
            for y in range(height):
                if self.map[b, f, x, y] == 0:  # empty field
                    self._render_field(x, y, color="white")
                else:  # wall
                    self._render_field(x, y, color="#555555")
                if np.all((x, y) == entrance) and f == 0:  # entrance
                    self._render_entrance(x, y)
                if np.all((x, y) == elevator):  # elevator
                    self._render_elevator(x, y)
                if np.all((b, f, x, y) == self.target) and not self.goal_reached:  # target
                    self._render_target(x, y)
                if np.all((x, y) == self.current_pos[2:4]):  # player
                    self._render_player(x, y)
                if self.render_option_history:
                    self._render_option_history_field(x, y)

        if self.option is not None or self.option_pos is not None:
            print(f"render termination based on: {'meta-policy' if self.option is None else 'option'}")
            self._render_termination_heatmap()
        elif self.action is not None:
            print("render action")
            self._render_action_heatmap()

    def _render_field(self, x_coord, y_coord, color=(255, 255, 255), alpha: float = 0, surface = None):
        if surface is None:
            surface = self.window
        x = MARGIN + BORDER_WIDTH + x_coord * (FIELD_SIZE + BORDER_WIDTH)
        y = MARGIN + BORDER_WIDTH + y_coord * (FIELD_SIZE + BORDER_WIDTH)
        pygame.draw.rect(surface, color, [x, y, FIELD_SIZE, FIELD_SIZE])

    def _render_player(self, x_coord, y_coord):
        x, y = self._get_field_center(x_coord, y_coord)
        color = TARGET_COLOR if self.goal_reached else PLAYER_COLOR
        pygame.draw.circle(self.window, "#FFFFFF", [x, y], 0.36 * FIELD_SIZE)
        pygame.draw.circle(self.window, color, [x, y], 0.33 * FIELD_SIZE)

    def _render_target(self, x_coord, y_coord):
        center = np.asarray(self._get_field_center(x_coord, y_coord))
        for i in range(5):
            angle = 2 * np.pi * i / 5
            rot = get_rotation_matrix(angle)
            shape = np.array([[-0.18, 0],
                              [0, -0.4],
                              [0.18, 0]]) * FIELD_SIZE
            polygon = center + np.dot(rot, shape.T).T
            pygame.draw.polygon(self.window, TARGET_COLOR, polygon)

    def _render_elevator(self, x_coord, y_coord):
        self._render_field(x_coord, y_coord, color=ELEVATOR_COLOR)
        center = np.asarray(self._get_field_center(x_coord, y_coord))
        shape = np.array([[-0.23, -0.08],
                          [0, -0.33],
                          [0.23, -0.08]]) * FIELD_SIZE
        up = center + shape
        down = center + shape * np.asarray([1, -1])
        pygame.draw.polygon(self.window, "white", up)
        pygame.draw.polygon(self.window, "white", down)

    def _render_entrance(self, x_coord, y_coord):
        self._render_field(x_coord, y_coord, color=ENTRANCE_COLOR)
        center = np.asarray(self._get_field_center(x_coord, y_coord))
        dir = self._get_center_direction(x_coord, y_coord) * 0.25 * FIELD_SIZE
        draw_arrow(self.window,
                   start_pos=center - dir,
                   end_pos=center + dir,
                   tip_width=int(0.4 * FIELD_SIZE),
                   color="white",
                   width=int(0.15 * FIELD_SIZE))

    def _render_option_history_field(self, x_coord, y_coord):
        b, f, x, y = self.current_pos
        history_option = self.option_history_map[b, f, x_coord, y_coord]
        if history_option != -1:
            draw_label(self.window,
                       position=[MARGIN + BORDER_WIDTH + x_coord * (FIELD_SIZE + BORDER_WIDTH),
                                 MARGIN + BORDER_WIDTH + y_coord * (FIELD_SIZE + BORDER_WIDTH)],
                       text=str(history_option),
                       font=pygame.font.SysFont('Source Code Pro', 20))

    def _render_buildings(self):
        margin_top = 2 * MARGIN + self.floor_shape[1] * (FIELD_SIZE + BORDER_WIDTH) + BORDER_WIDTH
        building_map_width = BUILDING_MARGIN + self.n_buildings * (BUILDING_WIDTH + BUILDING_MARGIN)
        margin_left = (self.window.get_width() - building_map_width) // 2

        # Ground line
        start_pos = [margin_left, margin_top + self.n_floors * (FLOOR_HEIGHT - BORDER_WIDTH)]
        end_pos = [margin_left + building_map_width, start_pos[1]]
        pygame.draw.line(self.window, "#000000", start_pos, end_pos, width=BORDER_WIDTH)

        # Buildings
        for b in range(self.n_buildings):
            for f in reversed(range(self.n_floors)):
                x = margin_left + BUILDING_MARGIN + b * (BUILDING_WIDTH + BUILDING_MARGIN)
                y = margin_top + (self.n_floors - f - 1) * (FLOOR_HEIGHT - BORDER_WIDTH)
                w = BUILDING_WIDTH
                h = FLOOR_HEIGHT
                pygame.draw.rect(self.window, "#000000", [x, y, w, h])
                if b == self.current_pos[0] and f == self.current_pos[1]:
                    floor_color = PLAYER_COLOR
                elif b == self.target[0] and f == self.target[1]:
                    floor_color = TARGET_COLOR
                else:
                    floor_color = "#FFFFFF"
                pygame.draw.rect(self.window, floor_color, [x + BORDER_WIDTH, y + BORDER_WIDTH,
                                                            w - 2 * BORDER_WIDTH, h - 2 * BORDER_WIDTH])

    def _get_center_direction(self, x, y):
        width, height = self.floor_shape
        diag_1 = x * (height - 1) / (width - 1)
        diag_2 = (height - 1) - x * (height - 1) / (width - 1)
        if y < diag_1:
            if y < diag_2:
                return direction["south"]
            else:
                return direction["west"]
        else:
            if y > diag_2:
                return direction["north"]
            else:
                return direction["east"]

    def _get_field_center(self, x_coord, y_coord):
        x = MARGIN + BORDER_WIDTH + x_coord * (FIELD_SIZE + BORDER_WIDTH) + FIELD_SIZE // 2
        y = MARGIN + BORDER_WIDTH + y_coord * (FIELD_SIZE + BORDER_WIDTH) + FIELD_SIZE // 2
        return x, y

    def _surrounded_by_walls(self, x, y, floor_map):
        position = np.asarray([x, y])
        for d in list(direction.values()):
            x_adjacent, y_adjacent = position + d
            if 0 <= x_adjacent < self.floor_shape[0] and 0 <= y_adjacent < self.floor_shape[1]:
                if not floor_map[x_adjacent, y_adjacent]:
                    return False
        return True

    def _display_frame(self):
        pygame.display.flip()
        pygame.event.pump()

    def get_keys_to_action(self):
        return {
            (pygame.K_w,): 0,  # north
            (pygame.K_d,): 1,  # east
            (pygame.K_s,): 2,  # south
            (pygame.K_a,): 3,  # west
            (pygame.K_EQUALS,): 4,  # next
            (pygame.K_MINUS,): 5,  # prev
        }

    def get_action_meanings(self) -> Sequence[str]:
        action_names = [a.name for a in Action]
        return action_names

    def _render_termination_heatmap(self):
        color = (0, 90, 169)

        # Init overlay surface
        overlay_surface = pygame.Surface(self.window.get_size(), pygame.SRCALPHA)

        # termination_tensor = th.tensor([1], dtype=th.float32, device="cuda")
        termination_tensor = th.tensor([1], dtype=th.float32)

        orig_pos = self.current_pos.copy()
        for x in range(self.floor_shape[0]):
            for y in range(self.floor_shape[1]):
                if not self.map[orig_pos[0], orig_pos[1], x, y]:  # if no wall
                    self.current_pos[2:4] = [x, y]
                    obs = self._get_observation()
                    # obs_norm = th.tensor(self.vec_norm.normalize_obs(obs), device="cuda").unsqueeze(0)
                    if self.vec_norm is not None:
                        obs_norm = th.tensor(self.vec_norm.normalize_obs(obs)).unsqueeze(0)
                    else:
                        obs_norm = th.tensor(obs).unsqueeze(0)

                    if self.option_pos is not None and self.policy is not None:
                        # use meta-policy termination probabilities
                        _, termination_probs = self.policy.meta_policy.get_terminations(obs_norm)
                        termination_probs = termination_probs[0, self.option_pos]
                    else:
                        # use option termination probabilities
                        termination_logp, _ = self.option.evaluate_terminations(obs_norm, termination_tensor)[0]
                        termination_probs = th.exp(termination_logp)
                    # alpha = int(200 * np.exp(termination_logp[0].cpu().detach().numpy()))
                    alpha = int(200 * termination_probs.cpu().detach().numpy())
                    self._render_field(x, y, color=(*color, alpha), surface=overlay_surface)
        self.current_pos = orig_pos

        # Apply overlay to screen
        self.window.blit(overlay_surface, (0, 0))

    def _render_action_heatmap(self):
        color = (230, 0, 26)

        # Init overlay surface
        overlay_surface = pygame.Surface(self.window.get_size(), pygame.SRCALPHA)

        # action_tensor = th.tensor([self.action], device="cuda")
        action_tensor = th.tensor([self.action])

        orig_pos = self.current_pos.copy()
        for x in range(self.floor_shape[0]):
            for y in range(self.floor_shape[1]):
                if not self.map[orig_pos[0], orig_pos[1], x, y]:  # if no wall
                    self.current_pos[2:4] = [x, y]
                    obs = self._get_observation()
                    # obs_norm = th.tensor(self.vec_norm.normalize_obs(obs), device="cuda").unsqueeze(0)
                    if self.vec_norm is not None:
                        obs_norm = th.tensor(self.vec_norm.normalize_obs(obs)).unsqueeze(0)
                    else:
                        obs_norm = th.tensor(obs).unsqueeze(0)
                    _, action_logp, _ = self.policy.evaluate_actions(obs_norm, action_tensor)
                    alpha = int(200 * np.exp(action_logp[0].cpu().detach().numpy()))
                    self._render_field(x, y, color=(*color, alpha), surface=overlay_surface)
        self.current_pos = orig_pos

        # Apply overlay to screen
        self.window.blit(overlay_surface, (0, 0))

    def register_current_option(self, option_id):
        self.option_history_map[tuple(self.current_pos)] = option_id

    def set_render_option_history(self, value: bool):
        self.render_option_history = value

    def render_termination_heatmap_of_option(self, option, vec_norm=None):
        self.option = option
        if option is not None and option.vec_norm is not None:
            vec_norm = option.vec_norm
        self.vec_norm = vec_norm

    def render_termination_heatmap_by_policy(self, policy, option_pos, vec_norm=None):
        # import ipdb; ipdb.set_trace()
        self.policy = policy
        self.option_pos = option_pos
        self.vec_norm = vec_norm

    def render_action_heatmap_for_policy(self, action, policy=None, vec_norm=None):
        self.policy = policy
        self.action = action
        self.vec_norm = vec_norm


def get_rotation_matrix(rad: float):
    return np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
