import torch as th
from torch import nn

from nsfr.fol.logic import Term
from logic.valuation.base import ValuationModule, ValuationFunction
from envs.meeting_room import PLAYER_VIEW_SIZE


class MeetingRoomValuationModule(ValuationModule):
    def init_valuation_functions(self):
        layers = [
            NorthOfTarget(),
            EastOfTarget(),
            SouthOfTarget(),
            WestOfTarget(),
            AboveTarget(),
            BelowTarget(),
            LeftOfTargetBuilding(),
            RightOfTargetBuilding(),
            OnTargetFloor(),
            NotOnTargetFloor(),
            InTargetBuilding(),
            NotInTargetBuilding(),
            NorthOfElevator(),
            EastOfElevator(),
            SouthOfElevator(),
            WestOfElevator(),
            NorthOfEntrance(),
            EastOfEntrance(),
            SouthOfEntrance(),
            WestOfEntrance(),
            OnGroundFloor(),
            NotOnGroundFloor(),
            InElevator(),
            AtEntrance(),
            WallNorth(),
            WallEast(),
            WallSouth(),
            WallWest(),
        ]

        pred_name_to_val_fn = {}
        for val_fn in layers:
            pred_name_to_val_fn[val_fn.pred_name] = val_fn

        return nn.ModuleList(layers), pred_name_to_val_fn

    def ground_to_tensor(self, term: Term, zs: th.Tensor):
        if term.dtype.name == 'image':
            return zs
        elif term.dtype.name == 'local_view':
            return self._extract_local_view(zs)
        else:
            raise ValueError(f"Unrecognised term input type '{term.dtype}'.")

    def _extract_local_view(self, logic_state: th.Tensor):
        local_view_shape = [-1, PLAYER_VIEW_SIZE, PLAYER_VIEW_SIZE]
        local_view = logic_state[..., 9:]
        return th.reshape(local_view, local_view_shape)


class NorthOfTarget(ValuationFunction):
    def __init__(self):
        super().__init__("north_of_target")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_target_pos = logic_state[..., 0:4]
        is_north_of = rel_target_pos[..., 3] > 0
        return self.bool2probs(is_north_of)


class EastOfTarget(ValuationFunction):
    def __init__(self):
        super().__init__("east_of_target")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_target_pos = logic_state[..., 0:4]
        is_east_of = rel_target_pos[..., 2] < 0
        return self.bool2probs(is_east_of)


class SouthOfTarget(ValuationFunction):
    def __init__(self):
        super().__init__("south_of_target")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_target_pos = logic_state[..., 0:4]
        is_south_of = rel_target_pos[..., 3] < 0
        return self.bool2probs(is_south_of)


class WestOfTarget(ValuationFunction):
    def __init__(self):
        super().__init__("west_of_target")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_target_pos = logic_state[..., 0:4]
        is_west_of = rel_target_pos[..., 2] > 0
        return self.bool2probs(is_west_of)


class BelowTarget(ValuationFunction):
    def __init__(self):
        super().__init__("below_target")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_target_pos_h = logic_state[..., 1]
        return self.bool2probs(rel_target_pos_h > 0)


class AboveTarget(ValuationFunction):
    def __init__(self):
        super().__init__("above_target")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_target_pos_h = logic_state[..., 1]
        return self.bool2probs(rel_target_pos_h < 0)


class LeftOfTargetBuilding(ValuationFunction):
    def __init__(self):
        super().__init__("left_of_target_building")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_target_pos_b = logic_state[..., 0]
        return self.bool2probs(rel_target_pos_b > 0)


class RightOfTargetBuilding(ValuationFunction):
    def __init__(self):
        super().__init__("right_of_target_building")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_target_pos_b = logic_state[..., 0]
        return self.bool2probs(rel_target_pos_b < 0)


class OnTargetFloor(ValuationFunction):
    def __init__(self):
        super().__init__("on_target_floor")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_target_pos_bh = logic_state[..., 0:2]
        on_target_floor = th.all(rel_target_pos_bh == 0, dim=-1)
        return self.bool2probs(on_target_floor)


class NotOnTargetFloor(ValuationFunction):
    def __init__(self):
        super().__init__("not_on_target_floor")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_target_pos_bh = logic_state[..., 0:2]
        on_target_floor = th.all(rel_target_pos_bh == 0, dim=-1)
        return self.bool2probs(~on_target_floor)


class InTargetBuilding(ValuationFunction):
    def __init__(self):
        super().__init__("in_target_building")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_target_pos_b = logic_state[..., 0]
        return self.bool2probs(rel_target_pos_b == 0)


class NotInTargetBuilding(ValuationFunction):
    def __init__(self):
        super().__init__("not_in_target_building")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_target_pos_b = logic_state[..., 0]
        return self.bool2probs(th.logical_not(rel_target_pos_b == 0))


class NorthOfElevator(ValuationFunction):
    def __init__(self):
        super().__init__("north_of_elevator")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_elevator_y = logic_state[..., 5]
        return self.bool2probs(rel_elevator_y > 0)


class EastOfElevator(ValuationFunction):
    def __init__(self):
        super().__init__("east_of_elevator")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_elevator_x = logic_state[..., 4]
        return self.bool2probs(rel_elevator_x < 0)


class SouthOfElevator(ValuationFunction):
    def __init__(self):
        super().__init__("south_of_elevator")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_elevator_y = logic_state[..., 5]
        return self.bool2probs(rel_elevator_y < 0)


class WestOfElevator(ValuationFunction):
    def __init__(self):
        super().__init__("west_of_elevator")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_elevator_x = logic_state[..., 4]
        return self.bool2probs(rel_elevator_x > 0)


class NorthOfEntrance(ValuationFunction):
    def __init__(self):
        super().__init__("north_of_entrance")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_entrance_y = logic_state[..., 7]
        return self.bool2probs(rel_entrance_y > 0)


class EastOfEntrance(ValuationFunction):
    def __init__(self):
        super().__init__("east_of_entrance")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_entrance_x = logic_state[..., 6]
        return self.bool2probs(rel_entrance_x < 0)


class SouthOfEntrance(ValuationFunction):
    def __init__(self):
        super().__init__("south_of_entrance")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_entrance_y = logic_state[..., 7]
        return self.bool2probs(rel_entrance_y < 0)


class WestOfEntrance(ValuationFunction):
    def __init__(self):
        super().__init__("west_of_entrance")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_entrance_x = logic_state[..., 6]
        return self.bool2probs(rel_entrance_x > 0)


class OnGroundFloor(ValuationFunction):
    def __init__(self):
        super().__init__("on_ground_floor")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        floor_no = logic_state[..., 8]
        return self.bool2probs(floor_no == 0)


class NotOnGroundFloor(ValuationFunction):
    def __init__(self):
        super().__init__("not_on_ground_floor")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        floor_no = logic_state[..., 8]
        return self.bool2probs(th.logical_not(floor_no == 0))


class InElevator(ValuationFunction):
    def __init__(self):
        super().__init__("in_elevator")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_elevator_pos = logic_state[..., 4:6]
        in_elevator = th.all(rel_elevator_pos == 0, dim=-1)
        return self.bool2probs(in_elevator)


class AtEntrance(ValuationFunction):
    def __init__(self):
        super().__init__("at_entrance")

    def forward(self, logic_state: th.Tensor) -> th.Tensor:
        rel_entrance_pos = logic_state[..., 6:8]
        at_entrance = th.all(rel_entrance_pos == 0, dim=-1)
        return self.bool2probs(at_entrance)


class WallNorth(ValuationFunction):
    def __init__(self):
        super().__init__("wall_north")

    def forward(self, local_view: th.Tensor) -> th.Tensor:
        x = PLAYER_VIEW_SIZE // 2
        y = PLAYER_VIEW_SIZE // 2 - 1
        return th.Tensor(local_view[:, x, y] == 1)


class WallEast(ValuationFunction):
    def __init__(self):
        super().__init__("wall_east")

    def forward(self, local_view: th.Tensor) -> th.Tensor:
        x = PLAYER_VIEW_SIZE // 2 + 1
        y = PLAYER_VIEW_SIZE // 2
        return th.Tensor(local_view[:, x, y] == 1)


class WallSouth(ValuationFunction):
    def __init__(self):
        super().__init__("wall_south")

    def forward(self, local_view: th.Tensor) -> th.Tensor:
        x = PLAYER_VIEW_SIZE // 2
        y = PLAYER_VIEW_SIZE // 2 + 1
        return th.Tensor(local_view[:, x, y] == 1)


class WallWest(ValuationFunction):
    def __init__(self):
        super().__init__("wall_west")

    def forward(self, local_view: th.Tensor) -> th.Tensor:
        x = PLAYER_VIEW_SIZE // 2 - 1
        y = PLAYER_VIEW_SIZE // 2
        return th.Tensor(local_view[:, x, y] == 1)
