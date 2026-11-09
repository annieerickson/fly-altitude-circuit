from .paths import configure_matplotlib_cache

configure_matplotlib_cache()

import matplotlib as mpl
from . import config

def apply_style():
    mpl.rcParams.update(config.load("style")["rcparams"])


def add_canvas_padding(fig, padding):
    """Add margins in inches without changing data-axis dimensions or spacing."""
    if not padding:
        return fig
    left, right, bottom, top = (padding.get(side, 0) for side in ("left", "right", "bottom", "top"))
    if any(value < 0 for value in (left, right, bottom, top)):
        raise ValueError("Canvas padding must be nonnegative.")
    width, height = fig.get_size_inches()
    positions = [(axis, axis.get_position().bounds) for axis in fig.axes]
    new_width, new_height = width + left + right, height + bottom + top
    fig.set_size_inches(new_width, new_height)
    for axis, (x, y, w, h) in positions:
        axis.set_position(((x * width + left) / new_width,
                           (y * height + bottom) / new_height,
                           w * width / new_width, h * height / new_height))
    return fig
