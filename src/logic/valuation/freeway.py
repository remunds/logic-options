import torch as th
from torch import nn

from nsfr.valuation import ValuationModule, ValuationFunction


class FreewayValuationModule(ValuationModule):
    def init_valuation_functions(self):
        layers = [
            Type(),
            Closeby(),
            OnLeft(),
            OnRight(),
            SameRow(),
            AboveRow(),
            Top5Cars(),
            Bottom5Cars(),
        ]
        return layers

    def ground_to_tensor(self, term, zs):
        term_index = self.lang.term_index(term)

        import re
        pattern = 'obj([1-9][0-9]*)'
        # check if the constant name is the reserved style e.g. obj1
        result = re.match(pattern, term.name)

        if not result == None:
            # it is an object constant
            obj_id = result[1]
            obj_index = int(obj_id) - 1
            return zs[:, obj_index]
        elif term.dtype.name == 'image':
            return zs
        else:
            # other attributes
            return self.term_to_onehot(term, batch_size=zs.size(0))

    def term_to_onehot(self, term, batch_size):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.

            Return:
                The tensor representation of the input term.
        """
        if term.dtype.name == 'color':
            return self.to_onehot_batch(self.colors.index(term.name), len(self.colors), batch_size)
        elif term.dtype.name == 'shape':
            return self.to_onehot_batch(self.shapes.index(term.name), len(self.shapes), batch_size)
        elif term.dtype.name == 'material':
            return self.to_onehot_batch(self.materials.index(term.name), len(self.materials), batch_size)
        elif term.dtype.name == 'size':
            return self.to_onehot_batch(self.sizes.index(term.name), len(self.sizes), batch_size)
        elif term.dtype.name == 'side':
            return self.to_onehot_batch(self.sides.index(term.name), len(self.sides), batch_size)
        elif term.dtype.name == 'type':
            return self.to_onehot_batch(self.lang.term_index(term), len(self.lang.get_by_dtype_name(term.dtype.name)),
                                        batch_size)
        else:
            assert True, 'Invalid term: ' + str(term)

    def to_onehot_batch(self, i, length, batch_size):
        """Compute the one-hot encoding that is expanded to the batch size.
        """
        onehot = th.zeros(batch_size, length, ).to(self.device)
        onehot[:, i] = 1.0
        return onehot


class Type(ValuationFunction):
    def __init__(self):
        super().__init__("type")

    def forward(self, z: th.Tensor, a: th.Tensor) -> th.Tensor:
        z_type = z[:, 0:2]  # [1, 0, 0, 0] * [1.0, 0, 0, 0] .sum = 0.0  type(obj1, key):0.0
        prob = (a * z_type).sum(dim=1)
        return prob


class Closeby(ValuationFunction):
    def __init__(self):
        super().__init__("closeby")

    def forward(self, z_1: th.Tensor, z_2: th.Tensor) -> th.Tensor:
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        dis_x = abs(c_1[:, 0] - c_2[:, 0]) / 171
        dis_y = abs(c_1[:, 1] - c_2[:, 1]) / 171

        result = th.where((dis_x < 2.5) & (dis_y <= 0.1), 0.99, 0.1)

        return result


class OnLeft(ValuationFunction):
    def __init__(self):
        super().__init__("on_left")

    def forward(self, z_1: th.Tensor, z_2: th.Tensor):
        c_1 = z_1[:, -2]
        c_2 = z_2[:, -2]
        diff = c_2 - c_1
        result = th.where(diff > 0, 0.99, 0.01)
        return result


class OnRight(ValuationFunction):
    def __init__(self):
        super().__init__("on_right")

    def forward(self, z_1: th.Tensor, z_2: th.Tensor):
        c_1 = z_1[:, -2]
        c_2 = z_2[:, -2]
        diff = c_2 - c_1
        result = th.where(diff < 0, 0.99, 0.01)
        return result


class SameRow(ValuationFunction):
    def __init__(self):
        super().__init__("same_row")

    def forward(self, z_1: th.Tensor, z_2: th.Tensor):
        c_1 = z_1[:, -1]
        c_2 = z_2[:, -1]
        diff = abs(c_2 - c_1)
        result = th.where(diff < 6, 0.99, 0.01)
        return result


class AboveRow(ValuationFunction):
    def __init__(self):
        super().__init__("above_row")

    def forward(self, z_1: th.Tensor, z_2: th.Tensor):
        c_1 = z_1[:, -1]
        c_2 = z_2[:, -1]
        diff = c_2 - c_1
        result1 = th.where(diff < 23, 0.99, 0.01)
        result2 = th.where(diff > 4, 0.99, 0.01)
        return result1 * result2


class Top5Cars(ValuationFunction):
    def __init__(self):
        super().__init__("top5car")

    def forward(self, z_1: th.Tensor):
        y = z_1[:, -1]
        result = th.where(y > 100, 0.99, 0.01)
        return result


class Bottom5Cars(ValuationFunction):
    def __init__(self):
        super().__init__("bottom5car")

    def forward(self, z_1: th.Tensor):
        y = z_1[:, -1]
        result = th.where(y < 100, 0.99, 0.01)
        return result
