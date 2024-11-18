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
    return bool_to_probs(abs(player_y - obj_y) < 6)


def deeper_than(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is (significantly) 'deeper than' the object."""
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    return bool_to_probs(th.Tensor(player_y > obj_y + 4))


def higher_than(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is (significantly) 'higher than' the object."""
    player_y = player[..., 2]
    obj_y = obj[..., 2]
    return bool_to_probs(th.Tensor(player_y < obj_y - 4))


def close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    player_x = player[..., 1]
    player_y = player[..., 2]
    obj_x = obj[..., 1]
    obj_y = obj[..., 2]
    result = (abs(player_x - obj_x) < 32) & (abs(player_y - obj_y) < 32)
    return bool_to_probs(result)


def left_of(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is 'left of' the object."""
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    return bool_to_probs(th.Tensor(player_x < obj_x))


def right_of(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
    """True iff the player is 'right of' the object."""
    player_x = player[..., 1]
    obj_x = obj[..., 1]
    return bool_to_probs(th.Tensor(player_x > obj_x))


def oxygen_low(oxygen_bar: th.Tensor) -> th.Tensor:
    """True iff oxygen bar is below 16/64."""
    result = oxygen_bar[..., 1] < 16
    return bool_to_probs(result)


def test_predicate_global(global_state: th.Tensor) -> th.Tensor:
    result = global_state[..., 0, 2] < 100
    print("Global result is", result)
    return bool_to_probs(result)


def test_predicate_object(agent: th.Tensor) -> th.Tensor:
    result = agent[..., 2] < 100
    print("Object result is", result)
    return bool_to_probs(result)


def true_predicate(agent: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.tensor([True]))


def false_predicate(agent: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.tensor([False]))
