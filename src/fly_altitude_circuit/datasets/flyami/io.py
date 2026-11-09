from contextlib import contextmanager, ExitStack
import h5py
from ...config import load
from ...paths import data_root

def line_path(line, kind):                      # kind: "wb" or "movie"
    ds = load("datasets")["optogenetic_screen"]
    p = data_root() / ds["folder"] / ds["per_line"][kind].format(line=line)
    if not p.exists():
        raise FileNotFoundError(f"Required read-only screen input not found: {p}. Check config/paths.yaml.")
    return p

@contextmanager
def open_line(line, *, load_wb=False):
    """Open movie inputs read-only; optionally open pooled wingbeat traces.

    The first tuple entry is None unless load_wb is requested for the
    plot_diff_from_avg=False trace calculation.
    """
    with ExitStack() as stack:
        wb = stack.enter_context(h5py.File(line_path(line, "wb"), "r")) if load_wb else None
        movie = stack.enter_context(h5py.File(line_path(line, "movie"), "r"))
        yield wb, movie
