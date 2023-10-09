from typing import SupportsFloat, Any
from enum import Enum

import pygame
import numpy as np
from numpy.random import randint
from gymnasium import Env
from gymnasium.core import RenderFrame, ActType, ObsType

from utils.render import draw_arrow

MARGIN = 50
FIELD_SIZE = 40
FLOOR_HEIGHT = 40
BUILDING_WIDTH = 70
BORDER_WIDTH = 2

PLAYER_COLOR = "#FF9911"
TARGET_COLOR = "#11FF99"


class Action(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    UP = 4
    DOWN = 5
    ENTER = 6


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
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    map: dict[int, dict[str | int, Any]]
    current_pos: np.array
    target: np.array

    def __init__(self, n_buildings=4, n_floors=4, floor_shape=None, render_mode: str = "human"):
        if floor_shape is None:
            floor_shape = np.array([11, 11], dtype=int)
        else:
            floor_shape = np.asarray(floor_shape)
        self.render_mode = render_mode
        self.n_buildings = n_buildings
        self.n_floors = n_floors
        assert np.all(floor_shape[0] >= 5)
        self.floor_shape = floor_shape
        self.inside_building = True
        self.terminated = False
        if self.render_mode == "human":
            self._init_pygame()

    def _generate_map(self):
        self.map = dict()
        for b in range(self.n_buildings):
            elevator_position = randint(1, self.floor_shape - 1)
            entrance_position = self._generate_entrance_position()
            self.map[b] = {"elevator": elevator_position,
                           "entrance": entrance_position}
            for f in range(self.n_floors):
                floor_map = self._generate_floor_map()

                # Remove walls in special cases
                if f == 0:
                    entrance_x, entrance_y = entrance_position
                    floor_map[entrance_x, entrance_y] = 0
                    c_dir = self._get_center_direction(*entrance_position)
                    floor_map[entrance_x + c_dir[0], entrance_y + c_dir[1]] = 0
                elevator_x, elevator_y = elevator_position
                floor_map[elevator_x, elevator_y] = 0
                if self._surrounded_by_walls(*elevator_position, floor_map):
                    c_dir = self._get_center_direction(*elevator_position)
                    floor_map[elevator_x + c_dir[0], elevator_y + c_dir[1]] = 0

                self.map[b][f] = floor_map

    def _generate_floor_map(self):
        """Creates an array encoding the floor's walls. Walls are represented
        by 1, empty space by 0."""
        floor_map = np.zeros(self.floor_shape)

        # Surrounding walls
        floor_map[[0, -1]] = 1
        floor_map[:, [0, -1]] = 1

        # Vertical wall
        wall_v = randint(2, self.floor_shape[1] - 2)
        floor_map[wall_v] = 1

        # Horizontal wall
        wall_h = randint(2, self.floor_shape[0] - 2, size=2)
        floor_map[:wall_v, wall_h[0]] = 1
        floor_map[wall_v:, wall_h[1]] = 1

        # Doorways
        # north
        n_choices = list(range(1, np.min(wall_h)))
        door_pos = np.random.choice(n_choices)
        floor_map[wall_v, door_pos] = 0

        # east
        n_choices = list(range(wall_v + 1, self.floor_shape[0] - 1))
        door_pos = np.random.choice(n_choices)
        floor_map[door_pos, wall_h[1]] = 0

        # south
        n_choices = list(range(np.max(wall_h) + 1, self.floor_shape[1] - 1))
        door_pos = np.random.choice(n_choices)
        floor_map[wall_v, door_pos] = 0

        # west
        n_choices = list(range(1, wall_v))
        door_pos = np.random.choice(n_choices)
        floor_map[door_pos, wall_h[0]] = 0

        return floor_map

    def _generate_entrance_position(self):
        # Pick one of the four sides of the building (0 north, 1 east etc.)
        side = randint(0, 4)

        # Determine entrance position inside wall
        wall_length = self.floor_shape[0] if side % 2 else self.floor_shape[1]
        pos_in_wall = randint(1, wall_length - 1)

        # Return final entrance door position
        match side:
            case 0:
                x, y = 0, pos_in_wall
            case 1:
                x, y = pos_in_wall, self.floor_shape[1] - 1
            case 2:
                x, y = self.floor_shape[0] - 1, pos_in_wall
            case _:
                x, y = pos_in_wall, 0
        return np.array([x, y], dtype=int)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self._generate_map()
        self.current_pos = self._pick_random_position()
        self.target = self.current_pos.copy()
        while np.all(self.target == self.current_pos):
            self.target = self._pick_random_position()
        self.inside_building = True
        self.terminated = False

        return self._get_observation(), self._get_info()

    def _pick_random_position(self):
        b = randint(self.n_buildings)
        f = randint(self.n_floors)

        # Find valid position on floor, not occupied by a wall
        floor_map = self.map[b][f]
        while True:
            x, y = randint(1, self.floor_shape - 1, size=2)
            if not floor_map[x, y]:  # empty spot
                break

        return np.array([b, f, x, y])

    def _get_observation(self):
        b, f, x, y = self.current_pos
        floor_map = self.map[b][f]
        elevator = self.map[b]["elevator"]
        entrance = self.map[b]["entrance"]
        return self.current_pos, floor_map, self.target, elevator, entrance, self.inside_building

    def _get_info(self):
        return dict()

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._act(Action(action))
        target_reached = np.all(self.current_pos == self.target)
        obs = self._get_observation()
        reward = 1 if target_reached and not self.terminated else 0
        self.terminated |= target_reached
        truncated = False
        info = self._get_info()
        return obs, reward, self.terminated, truncated, info

    def _act(self, action: Action):
        b, f, x, y = self.current_pos
        elevator = self.map[b]["elevator"]
        entrance = self.map[b]["entrance"]

        if not self.inside_building:
            if action in [Action.EAST, Action.WEST]:
                if action == Action.EAST and b < self.n_buildings - 1:
                    self.current_pos[0] += 1
                elif action == Action.WEST and b > 0:
                    self.current_pos[0] -= 1
                entrance = self._get_current_entrance()
                self.current_pos[2:4] = entrance
                return

        elif action in [Action.NORTH, Action.EAST, Action.SOUTH, Action.WEST]:
            next_pos = None
            match action:
                case Action.NORTH: next_pos = self.current_pos[2:4] + direction["north"]
                case Action.EAST: next_pos = self.current_pos[2:4] + direction["east"]
                case Action.SOUTH: next_pos = self.current_pos[2:4] + direction["south"]
                case Action.WEST: next_pos = self.current_pos[2:4] + direction["west"]

            if self._is_valid_next_step(*next_pos):
                self.current_pos[2:4] = next_pos
            return

        match action:
            case Action.UP:
                if np.all(self.current_pos[2:4] == elevator) and f < self.n_floors - 1:
                    self.current_pos[1] += 1

            case Action.DOWN:
                if np.all(self.current_pos[2:4] == elevator) and f > 0:
                    self.current_pos[1] -= 1

            case Action.ENTER:
                if np.all(self.current_pos[2:4] == entrance) and f == 0:
                    self.inside_building = not self.inside_building

    def _is_valid_next_step(self, x, y):
        floor_map = self._get_current_floor_map()
        elevator = self._get_current_elevator()
        entrance = self._get_current_entrance()
        return 0 <= x < self.floor_shape[0] and \
            0 <= y < self.floor_shape[1] and \
            (not floor_map[x, y] or np.all([x, y] == elevator) or np.all([x, y] == entrance))

    def _get_current_floor_map(self):
        b, f, _, _ = self.current_pos
        return self.map[b][f]

    def _get_current_entrance(self):
        return self.map[self.current_pos[0]]["entrance"]

    def _get_current_elevator(self):
        return self.map[self.current_pos[0]]["elevator"]

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Meeting Room")
        self.window_shape = self.floor_shape * (FIELD_SIZE + BORDER_WIDTH) + 2 * BORDER_WIDTH + 2 * MARGIN
        self.window_shape[1] += self.n_floors * FLOOR_HEIGHT + MARGIN
        self.window = pygame.display.set_mode(self.window_shape)
        self.clock = pygame.time.Clock()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        self._render_frame()
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.window)
        else:
            self.clock.tick()
            self._display_frame()

    def _render_frame(self):
        self.window.fill((255, 255, 255))  # clear the entire window
        self._render_grid()
        self._render_buildings()

    def _render_grid(self):
        b, f, _, _ = self.current_pos
        floor_map = self._get_current_floor_map()
        height, width = floor_map.shape

        # Draw grid background
        grid_width = width * (FIELD_SIZE + BORDER_WIDTH) + BORDER_WIDTH
        grid_height = height * (FIELD_SIZE + BORDER_WIDTH) + BORDER_WIDTH
        pygame.draw.rect(self.window, (0, 0, 0), [MARGIN, MARGIN, grid_width, grid_height])

        entrance = self._get_current_entrance()
        elevator = self._get_current_elevator()

        # Draw all fields
        for x in range(width):
            for y in range(height):
                if floor_map[x, y] == 0:  # empty field
                    self._render_field(x, y, color="white")
                else:  # wall
                    self._render_field(x, y, color="#555555")
                if np.all((b, f, x, y) == self.target) and not self.terminated:  # target
                    self._render_target(x, y)
                if np.all((x, y) == entrance) and f == 0:  # entrance
                    self._render_entrance(x, y)
                if np.all((x, y) == elevator):  # elevator
                    self._render_elevator(x, y)
                if np.all((x, y) == self.current_pos[2:4]):  # player
                    self._render_player(x, y)

    def _render_field(self, x_coord, y_coord, color=(255, 255, 255)):
        x = MARGIN + BORDER_WIDTH + x_coord * (FIELD_SIZE + BORDER_WIDTH)
        y = MARGIN + BORDER_WIDTH + y_coord * (FIELD_SIZE + BORDER_WIDTH)
        pygame.draw.rect(self.window, color, [x, y, FIELD_SIZE, FIELD_SIZE])

    def _render_player(self, x_coord, y_coord):
        x, y = self._get_field_center(x_coord, y_coord)
        pygame.draw.circle(self.window, "#000000", [x, y], 0.35 * FIELD_SIZE)
        pygame.draw.circle(self.window, PLAYER_COLOR, [x, y], 0.3 * FIELD_SIZE)

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
        center = np.asarray(self._get_field_center(x_coord, y_coord))
        shape = np.array([[-0.45, -0.1],
                          [0, -0.45],
                          [0.45, -0.1]]) * FIELD_SIZE
        up = center + shape
        down = center + shape * np.asarray([1, -1])
        pygame.draw.polygon(self.window, "#FF2211", up)
        pygame.draw.polygon(self.window, "#FF2211", down)

    def _render_entrance(self, x_coord, y_coord):
        center = np.asarray(self._get_field_center(x_coord, y_coord))
        dir = self._get_center_direction(x_coord, y_coord) * 0.3 * FIELD_SIZE
        draw_arrow(self.window,
                   start_pos=center - dir,
                   end_pos=center + dir,
                   tip_width=int(0.4*FIELD_SIZE),
                   color="#11AAFF",
                   width=int(0.15 * FIELD_SIZE))

    def _render_buildings(self):
        margin_top = 2 * MARGIN + self.floor_shape[1] * (FIELD_SIZE + BORDER_WIDTH) + BORDER_WIDTH
        margin_left = MARGIN

        # Ground line
        start_pos = [margin_left - 30, margin_top + self.n_floors * (FLOOR_HEIGHT - BORDER_WIDTH)]
        end_pos = [margin_left + self.n_buildings * (BUILDING_WIDTH + 30), start_pos[1]]
        pygame.draw.line(self.window, "#000000", start_pos, end_pos, width=BORDER_WIDTH)

        # Buildings
        for b in range(self.n_buildings):
            for f in reversed(range(self.n_floors)):
                x = margin_left + b * (BUILDING_WIDTH + 30)
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
            (pygame.K_PLUS,): 4,  # up
            (pygame.K_MINUS,): 5,  # down
            (pygame.K_SPACE,): 6,  # enter
        }


def get_rotation_matrix(rad: float):
    return np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
