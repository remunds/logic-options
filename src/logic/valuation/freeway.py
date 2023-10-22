import torch as th
from torch import nn

from logic.valuation.valuation_modules import ValuationModule


class FreewayValuationModule(ValuationModule):
    def init_valuation_functions(self):
        layers = []
        vfs = {}  # pred name -> valuation function
        v_type = TypeValuationFunction()
        vfs['type'] = v_type
        layers.append(v_type)

        # TODO
        v_closeby = ClosebyValuationFunction(self.device)
        vfs['closeby'] = v_closeby
        # vfs['closeby'].load_state_dict(torch.load(
        #     '../nudge/weights/neural_predicates/closeby_pretrain.pt', map_location=device))
        # vfs['closeby'].eval()
        layers.append(v_closeby)
        # print('Pretrained  neural predicate closeby have been loaded!')

        v_on_left = OnLeftValuationFunction()
        vfs['on_left'] = v_on_left
        layers.append(v_on_left)

        v_on_right = OnRightValuationFunction()
        vfs['on_right'] = v_on_right
        layers.append(v_on_right)

        v_same_row = SameRowValuationFunction()
        vfs['same_row'] = v_same_row
        layers.append(v_same_row)

        v_above_row = AboveRowValuationFunction()
        vfs['above_row'] = v_above_row
        layers.append(v_above_row)

        v_top5cars = Top5CarsValuationFunction()
        vfs['top5car'] = v_top5cars
        layers.append(v_top5cars)

        v_bottom5cars = Bottom5CarsValuationFunction()
        vfs['bottom5car'] = v_bottom5cars
        layers.append(v_bottom5cars)

        return nn.ModuleList(layers), vfs

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


class TypeValuationFunction(nn.Module):
    """The function v_object-type
    type(obj1, agent):0.98
    type(obj2, enemy）：0.87
    """

    def __init__(self):
        super(TypeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [agent, key, door, enemy, x, y]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_type = z[:, 0:2]  # [1, 0, 0, 0] * [1.0, 0, 0, 0] .sum = 0.0  type(obj1, key):0.0
        prob = (a * z_type).sum(dim=1)
        return prob


class ClosebyValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self, device):
        super(ClosebyValuationFunction, self).__init__()
        self.device = device

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        dis_x = abs(c_1[:, 0] - c_2[:, 0]) / 171
        dis_y = abs(c_1[:, 1] - c_2[:, 1]) / 171

        result = th.where((dis_x < 2.5) & (dis_y <= 0.1), 0.99, 0.1)

        return result


class OnLeftValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(OnLeftValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2]
        c_2 = z_2[:, -2]
        diff = c_2 - c_1
        result = th.where(diff > 0, 0.99, 0.01)
        return result


class OnRightValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(OnRightValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args: x
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2]
        c_2 = z_2[:, -2]
        diff = c_2 - c_1
        result = th.where(diff < 0, 0.99, 0.01)
        return result


class SameRowValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(SameRowValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args: x
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -1]
        c_2 = z_2[:, -1]
        diff = abs(c_2 - c_1)
        result = th.where(diff < 6, 0.99, 0.01)
        return result


class AboveRowValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(AboveRowValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args: x
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -1]
        c_2 = z_2[:, -1]
        diff = c_2 - c_1
        result1 = th.where(diff < 23, 0.99, 0.01)
        result2 = th.where(diff > 4, 0.99, 0.01)
        return result1 * result2


class Top5CarsValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(Top5CarsValuationFunction, self).__init__()

    def forward(self, z_1):
        """
        Args: x
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        y = z_1[:, -1]
        result = th.where(y > 100, 0.99, 0.01)
        return result


class Bottom5CarsValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(Bottom5CarsValuationFunction, self).__init__()

    def forward(self, z_1):
        """
        Args: x
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        y = z_1[:, -1]
        result = th.where(y < 100, 0.99, 0.01)
        return result


class HaveKeyValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(HaveKeyValuationFunction, self).__init__()

    def forward(self, z):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        has_key = th.ones(z.size(dim=0))
        c = th.sum(z[:, :, 1], dim=1)
        result = has_key[:] - c[:]

        return result


class NotHaveKeyValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(NotHaveKeyValuationFunction, self).__init__()

    def forward(self, z):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        c = th.sum(z[:, :, 1], dim=1)
        result = c[:]

        return result


class SafeValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(SafeValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        dis_x = abs(c_1[:, 0] - c_2[:, 0])
        result = th.where(dis_x > 2, 0.99, 0.01)
        return result
