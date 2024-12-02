from __future__ import annotations

import numpy as np
from sty import Style, RgbFg, fg, ef, rs

"""Colors and formatting for console text"""

fg.orange = Style(RgbFg(255, 130, 50))
fg.red = Style(RgbFg(255, 30, 30))
fg.green = Style(RgbFg(90, 255, 30))
fg.blue = Style(RgbFg(30, 90, 255))
fg.light_blue = Style(RgbFg(100, 170, 255))


def red(text):
    return fg.red + text + fg.rs


def orange(text):
    return fg.orange + text + fg.rs


def yellow(text):
    return "\033[93m" + text + "\033[0m"


def green(text):
    return fg.green + text + fg.rs


def blue(text):
    return fg.blue + text + fg.rs


def light_blue(text):
    return fg.light_blue + text + fg.rs


def bold(text):
    return "\033[1m" + text + "\033[0m"


def it(text):  # italic
    return "\033[3m" + text + "\033[0m"


def ul(text):  # underline
    return "\033[4m" + text + "\033[0m"


def num2text(num):
    if num == 0:
        return "0"
    elif np.abs(num) < 1:
        return "%.2f" % num
    elif np.abs(num) < 10 and num % 1 != 0:
        return "%.1f" % num
    elif np.abs(num) < 1000:
        return "%.0f" % num
    elif np.abs(num) < 10000:
        thousands = num / 1000
        return "%.1fK" % thousands
    elif np.abs(num) < 1e6:
        thousands = num / 1000
        return "%.0fK" % thousands
    elif np.abs(num) < 1e7:
        millions = num / 1e6
        return "%.1fM" % millions
    else:
        millions = num / 1e6
        return "%.0fM" % millions


def sec2hhmmss(s):
    m = s // 60
    h = m // 60
    return "%d:%02d:%02d h" % (h, m % 60, s % 60)
