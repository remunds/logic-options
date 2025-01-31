import torch as th

from nsfr.utils.common import bool_to_probs


# def visible(obj: th.Tensor) -> th.Tensor:
#     result = obj[..., 0] == 1
#     return bool_to_probs(result)


# def facing_left(player: th.Tensor) -> th.Tensor:
#     result = player[..., 3] == 12
#     return bool_to_probs(result)


# def facing_right(player: th.Tensor) -> th.Tensor:
#     result = player[..., 3] == 4
#     return bool_to_probs(result)


# def same_depth(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
#     player_y = player[..., 2]
#     obj_y = obj[..., 2]
#     return bool_to_probs(abs(player_y - obj_y) < 6)


# def deeper_than(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
#     """True iff the player is (significantly) 'deeper than' the object."""
#     player_y = player[..., 2]
#     obj_y = obj[..., 2]
#     return bool_to_probs(th.Tensor(player_y > obj_y + 4))


# def higher_than(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
#     """True iff the player is (significantly) 'higher than' the object."""
#     player_y = player[..., 2]
#     obj_y = obj[..., 2]
#     return bool_to_probs(th.Tensor(player_y < obj_y - 4))


# def close_by(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
#     player_x = player[..., 1]
#     player_y = player[..., 2]
#     obj_x = obj[..., 1]
#     obj_y = obj[..., 2]
#     result = (abs(player_x - obj_x) < 32) & (abs(player_y - obj_y) < 32)
#     return bool_to_probs(result)


# def left_of(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
#     """True iff the player is 'left of' the object."""
#     player_x = player[..., 1]
#     obj_x = obj[..., 1]
#     return bool_to_probs(th.Tensor(player_x < obj_x))


# def right_of(player: th.Tensor, obj: th.Tensor) -> th.Tensor:
#     """True iff the player is 'right of' the object."""
#     player_x = player[..., 1]
#     obj_x = obj[..., 1]
#     return bool_to_probs(th.Tensor(player_x > obj_x))


# def oxygen_low(oxygen_bar: th.Tensor) -> th.Tensor:
#     """True iff oxygen bar is below 16/64."""
#     result = oxygen_bar[..., 1] < 16
#     return bool_to_probs(result)

# def oxygen_high_or_med(oxygen_bar: th.Tensor) -> th.Tensor:
#     """True iff oxygen bar is above 16/64."""
#     result = oxygen_bar[..., 1] >= 16
#     return bool_to_probs(result)

# # check all 6 divers for visibility
# def diver_available(collected_diver1: th.Tensor, collected_diver2: th.Tensor, collected_diver3: th.Tensor, collected_diver4: th.Tensor, collected_diver5: th.Tensor, collected_diver6: th.Tensor) -> th.Tensor:
#     """True iff at least one collected diver symbol is not visible (and thus available)."""
#     # result = visible(collected_diver1) & visible(collected_diver2) & visible(collected_diver3) & visible(collected_diver4) & visible(collected_diver5) & visible(collected_diver6) 
#     # simply check visibility
#     result = collected_diver1[..., 0] == 0 or collected_diver2[..., 0] == 0 or collected_diver3[..., 0] == 0 or collected_diver4[..., 0] == 0 or collected_diver5[..., 0] == 0 or collected_diver6[..., 0] == 0
#     return bool_to_probs(result)


# def test_predicate_global(global_state: th.Tensor) -> th.Tensor:
#     result = global_state[..., 0, 2] < 100
#     print("Global result is", result)
#     return bool_to_probs(result)


# def test_predicate_object(agent: th.Tensor) -> th.Tensor:
#     result = agent[..., 2] < 100
#     print("Object result is", result)
#     return bool_to_probs(result)


# def true_predicate(agent: th.Tensor) -> th.Tensor:
#     return bool_to_probs(th.tensor([True]))


# def false_predicate(agent: th.Tensor) -> th.Tensor:
#     return bool_to_probs(th.tensor([False]))

def oxygen_low(obs: th.Tensor) -> th.Tensor:
    # obs has shape (N, 4, 90) (N is batch_size, 4 is window size)
    # we have 45 objects (hud), 2 features each
    oxygen_bar_idx = (1+12+12+4+4+1+1) * 2
    # oxygen_value = obs[:, -1].squeeze()[oxygen_bar_idx]
    oxygen_value = obs[:, -1, oxygen_bar_idx]
    return bool_to_probs((oxygen_value <= 16) & (oxygen_value > 0))

def not_oxygen_low(obs: th.Tensor) -> th.Tensor:
    return 1 - oxygen_low(obs)

def divers_visible(obs: th.Tensor) -> th.Tensor:
    diver_idx_start = (1+12+12)*2
    diver_idx_end = diver_idx_start + 4*2
    divers = obs[:, -1, diver_idx_start:diver_idx_end]
    return bool_to_probs(th.any(divers > 0))

def not_divers_visible(obs: th.Tensor) -> th.Tensor:
    return 1 - divers_visible(obs)

def enemies_visible(obs: th.Tensor) -> th.Tensor:
    enemy_idx_start = 1 * 2
    enemy_idx_end = enemy_idx_start + (12+12)*2
    enemies = obs[:, -1, enemy_idx_start:enemy_idx_end]
    return bool_to_probs(th.any(enemies > 0))

def not_enemies_visible(obs: th.Tensor) -> th.Tensor:
    return 1 - enemies_visible(obs)

def full_divers(obs: th.Tensor) -> th.Tensor:
    collected_diver_idx_start = (1+12+12+4+4+1+1)*2 + 1
    collected_diver_idx_end = collected_diver_idx_start + 6*2
    collected_divers = obs[:, -1, collected_diver_idx_start:collected_diver_idx_end]
    return bool_to_probs(th.all(collected_divers > 0))    

def not_full_divers(obs: th.Tensor) -> th.Tensor:
    return 1 - full_divers(obs)