import torch as th

from nsfr.utils.common import bool_to_probs
from logic_options.envs.meeting_room import PLAYER_VIEW_SIZE


def north_of_target(logic_state: th.Tensor) -> th.Tensor:
    rel_target_pos = logic_state[..., 0:4]
    is_north_of = rel_target_pos[..., 3] > 0
    return bool_to_probs(is_north_of)


def east_of_target(logic_state: th.Tensor) -> th.Tensor:
    rel_target_pos = logic_state[..., 0:4]
    is_east_of = rel_target_pos[..., 2] < 0
    return bool_to_probs(is_east_of)


def south_of_target(logic_state: th.Tensor) -> th.Tensor:
    rel_target_pos = logic_state[..., 0:4]
    is_south_of = rel_target_pos[..., 3] < 0
    return bool_to_probs(is_south_of)


def west_of_target(logic_state: th.Tensor) -> th.Tensor:
    rel_target_pos = logic_state[..., 0:4]
    is_west_of = rel_target_pos[..., 2] > 0
    return bool_to_probs(is_west_of)


def below_target(logic_state: th.Tensor) -> th.Tensor:
    rel_target_pos_h = logic_state[..., 1]
    return bool_to_probs(rel_target_pos_h > 0)


def above_target(logic_state: th.Tensor) -> th.Tensor:
    rel_target_pos_h = logic_state[..., 1]
    return bool_to_probs(rel_target_pos_h < 0)


def left_of_target_building(logic_state: th.Tensor) -> th.Tensor:
    rel_target_pos_b = logic_state[..., 0]
    return bool_to_probs(rel_target_pos_b > 0)


def right_of_target_building(logic_state: th.Tensor) -> th.Tensor:
    rel_target_pos_b = logic_state[..., 0]
    return bool_to_probs(rel_target_pos_b < 0)


def on_target_floor(logic_state: th.Tensor) -> th.Tensor:
    rel_target_pos_bh = logic_state[..., 0:2]
    on_target_floor = th.all(rel_target_pos_bh == 0, dim=-1)
    return bool_to_probs(on_target_floor)


def not_on_target_floor(logic_state: th.Tensor) -> th.Tensor:
    rel_target_pos_bh = logic_state[..., 0:2]
    on_target_floor = th.all(rel_target_pos_bh == 0, dim=-1)
    return bool_to_probs(~on_target_floor)


def in_target_building(logic_state: th.Tensor) -> th.Tensor:
    rel_target_pos_b = logic_state[..., 0]
    return bool_to_probs(rel_target_pos_b == 0)


def not_in_target_building(logic_state: th.Tensor) -> th.Tensor:
    rel_target_pos_b = logic_state[..., 0]
    return bool_to_probs(th.logical_not(rel_target_pos_b == 0))


def north_of_elevator(logic_state: th.Tensor) -> th.Tensor:
    rel_elevator_y = logic_state[..., 5]
    return bool_to_probs(rel_elevator_y > 0)


def east_of_elevator(logic_state: th.Tensor) -> th.Tensor:
    rel_elevator_x = logic_state[..., 4]
    return bool_to_probs(rel_elevator_x < 0)


def south_of_elevator(logic_state: th.Tensor) -> th.Tensor:
    rel_elevator_y = logic_state[..., 5]
    return bool_to_probs(rel_elevator_y < 0)


def west_of_elevator(logic_state: th.Tensor) -> th.Tensor:
    rel_elevator_x = logic_state[..., 4]
    return bool_to_probs(rel_elevator_x > 0)


def north_of_entrance(logic_state: th.Tensor) -> th.Tensor:
    rel_entrance_y = logic_state[..., 7]
    return bool_to_probs(rel_entrance_y > 0)


def east_of_entrance(logic_state: th.Tensor) -> th.Tensor:
    rel_entrance_x = logic_state[..., 6]
    return bool_to_probs(rel_entrance_x < 0)


def south_of_entrance(logic_state: th.Tensor) -> th.Tensor:
    rel_entrance_y = logic_state[..., 7]
    return bool_to_probs(rel_entrance_y < 0)


def west_of_entrance(logic_state: th.Tensor) -> th.Tensor:
    rel_entrance_x = logic_state[..., 6]
    return bool_to_probs(rel_entrance_x > 0)


def on_ground_floor(logic_state: th.Tensor) -> th.Tensor:
    floor_no = logic_state[..., 8]
    return bool_to_probs(floor_no == 0)


def not_on_ground_floor(logic_state: th.Tensor) -> th.Tensor:
    floor_no = logic_state[..., 8]
    return bool_to_probs(th.logical_not(floor_no == 0))


def in_elevator(logic_state: th.Tensor) -> th.Tensor:
    rel_elevator_pos = logic_state[..., 4:6]
    in_elevator = th.all(rel_elevator_pos == 0, dim=-1)
    return bool_to_probs(in_elevator)


def at_entrance(logic_state: th.Tensor) -> th.Tensor:
    rel_entrance_pos = logic_state[..., 6:8]
    at_entrance = th.all(rel_entrance_pos == 0, dim=-1)
    return bool_to_probs(at_entrance)


def wall_north(local_view: th.Tensor) -> th.Tensor:
    x = PLAYER_VIEW_SIZE // 2
    y = PLAYER_VIEW_SIZE // 2 - 1
    return th.Tensor(local_view[:, x, y] == 1)


def north_free(local_view: th.Tensor) -> th.Tensor:
    x = PLAYER_VIEW_SIZE // 2
    y = PLAYER_VIEW_SIZE // 2 - 1
    return th.Tensor(local_view[:, x, y] == 0)


def wall_east(local_view: th.Tensor) -> th.Tensor:
    x = PLAYER_VIEW_SIZE // 2 + 1
    y = PLAYER_VIEW_SIZE // 2
    return th.Tensor(local_view[:, x, y] == 1)


def east_free(local_view: th.Tensor) -> th.Tensor:
    x = PLAYER_VIEW_SIZE // 2 + 1
    y = PLAYER_VIEW_SIZE // 2
    return th.Tensor(local_view[:, x, y] == 0)


def wall_south(local_view: th.Tensor) -> th.Tensor:
    x = PLAYER_VIEW_SIZE // 2
    y = PLAYER_VIEW_SIZE // 2 + 1
    return th.Tensor(local_view[:, x, y] == 1)


def south_free(local_view: th.Tensor) -> th.Tensor:
    x = PLAYER_VIEW_SIZE // 2
    y = PLAYER_VIEW_SIZE // 2 + 1
    return th.Tensor(local_view[:, x, y] == 0)


def wall_west(local_view: th.Tensor) -> th.Tensor:
    x = PLAYER_VIEW_SIZE // 2 - 1
    y = PLAYER_VIEW_SIZE // 2
    return th.Tensor(local_view[:, x, y] == 1)


def west_free(local_view: th.Tensor) -> th.Tensor:
    x = PLAYER_VIEW_SIZE // 2 - 1
    y = PLAYER_VIEW_SIZE // 2
    return th.Tensor(local_view[:, x, y] == 0)
