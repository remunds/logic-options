import torch as th


def closeby(z_1: th.Tensor, z_2: th.Tensor) -> th.Tensor:
    c_1 = z_1[:, -2:]
    c_2 = z_2[:, -2:]

    dis_x = abs(c_1[:, 0] - c_2[:, 0]) / 171
    dis_y = abs(c_1[:, 1] - c_2[:, 1]) / 171

    result = th.where((dis_x < 2.5) & (dis_y <= 0.1), 0.99, 0.1)

    return result


def on_left(z_1: th.Tensor, z_2: th.Tensor):
    c_1 = z_1[:, -2]
    c_2 = z_2[:, -2]
    diff = c_2 - c_1
    result = th.where(diff > 0, 0.99, 0.01)
    return result


def on_right(z_1: th.Tensor, z_2: th.Tensor):
    c_1 = z_1[:, -2]
    c_2 = z_2[:, -2]
    diff = c_2 - c_1
    result = th.where(diff < 0, 0.99, 0.01)
    return result


def same_row(z_1: th.Tensor, z_2: th.Tensor):
    c_1 = z_1[:, -1]
    c_2 = z_2[:, -1]
    diff = abs(c_2 - c_1)
    result = th.where(diff < 6, 0.99, 0.01)
    return result


def above_row(z_1: th.Tensor, z_2: th.Tensor):
    c_1 = z_1[:, -1]
    c_2 = z_2[:, -1]
    diff = c_2 - c_1
    result1 = th.where(diff < 23, 0.99, 0.01)
    result2 = th.where(diff > 4, 0.99, 0.01)
    return result1 * result2


def top5car(z_1: th.Tensor):
    y = z_1[:, -1]
    result = th.where(y > 100, 0.99, 0.01)
    return result


def bottom5car(z_1: th.Tensor):
    y = z_1[:, -1]
    result = th.where(y < 100, 0.99, 0.01)
    return result
