import itertools
from typing import List

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

green = "#2ecc71"
blue = "#3498db"
purple = "#9b59b6"
yellow = "#f1c40f"
orange = "#e67e22"
red = "#e74c3c"
grey = "#ecf0f1"
default_colors = [green, blue, purple, yellow, orange, red, grey]
default_colors_cycle = itertools.cycle(default_colors)


def color_scale(fr: str, to: str, n: int) -> List[str]:
    """Interpolate hex colors between fr and to

    Args:
        fr (str): from hex color
        to (str): to hex color
        n (int): number of steps

    Example:
        >>> color_scale(green, blue, 5)
        ['#2ecc71', '#30bf8b', '#31b2a6', '#32a5c0', '#3498db']

    Returns:
        List[str]: list of hex string with colors from `fr` to `to`
    """
    fr = np.array(matplotlib.colors.to_rgba(fr))
    to = np.array(matplotlib.colors.to_rgba(to))
    steps = np.arange(n)[:, np.newaxis] / (n - 1)
    cs = fr + (to - fr) * steps
    return [matplotlib.colors.to_hex(c) for c in cs]
