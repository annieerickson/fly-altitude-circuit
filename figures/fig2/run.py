"""Figures 2B/C: aerodynamic force trace and wing/force overlay."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from robofly_workflow import run


def main(argv=None):
    run(argv, HERE / "params.yaml", HERE / "outputs", {"fig2B": "b", "fig2C": "c"})


if __name__ == "__main__":
    main()
