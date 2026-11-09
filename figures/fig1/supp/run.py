"""Generate Figures S1/S2, related to Figure 1."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parent))

from fly_altitude_circuit.runner import run_figure_panels
from screen_panels import plot_avg_traces_with_changes, plot_key_point_summary

PANELS = {
    "figS1_top": ("s1_top", plot_avg_traces_with_changes, "axes"),
    "figS1_bottom": ("s1_bottom", plot_avg_traces_with_changes, "axes"),
    "figS2": ("s2", plot_key_point_summary, "key_point_axes"),
}


def main(argv=None):
    run_figure_panels(argv, HERE / "params.yaml", HERE / "outputs", PANELS)


if __name__ == "__main__":
    main()
