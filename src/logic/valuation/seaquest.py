import torch as th

from nsfr.utils.common import bool_to_probs


def visible(obj: th.Tensor) -> th.Tensor:
    result = obj[..., 0] == 1
    return bool_to_probs(result)


def facing_left(player: th.Tensor) -> th.Tensor:
    result = player[..., 3] == 12
    return bool_to_probs(result)


def facing_right(player: th.Tensor) -> th.Tensor:
    result = player[..., 3] == 4
    return bool_to_probs(result)


def same_depth(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    return bool_to_probs(abs(player_y - obj_y) < 10)


def deeper_than(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    return bool_to_probs(th.Tensor(player_y > obj_y))


def higher_than(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    return bool_to_probs(th.Tensor(player_y < obj_y))


def close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    player_y = player[..., 2]
    obj_x = obj[..., 1]
    obj_y = obj[..., 2]
    result = (abs(player_x - obj_x) < 10) & (abs(player_y - obj_y) < 10)
    return bool_to_probs(result)


def on_left(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    return bool_to_probs(th.Tensor(player_x < obj_x))


def on_right(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    return bool_to_probs(th.Tensor(player_x > obj_x))


def oxygen_low(oxygen_bar: th.Tensor) -> th.Tensor:
    """True iff oxygen bar is below 10%."""
    result = oxygen_bar[..., 1] < 0.1
    return bool_to_probs(result)
