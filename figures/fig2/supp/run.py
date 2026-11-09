"""Figure S4 force components, related to Figure 2."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parent))

from robofly_workflow import run


def main(argv=None):
    run(argv, HERE / "params.yaml", HERE / "outputs", {"figS4": "s4"})


if __name__ == "__main__":
    main()
