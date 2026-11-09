"""Generate Figure 1 screen and activation panels."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from fly_altitude_circuit.runner import run_figure_panels
from screen_panels import plot_avg_traces_with_changes, plot_key_point_summary
from activation_workflow import run_activation

PANELS = {
    "fig1D_top": ("d_top", plot_avg_traces_with_changes, "axes"),
    "fig1D_bottom": ("d_bottom", plot_key_point_summary, "key_point_axes"),
}

WORKFLOWS = {
    "fig1F": ("f", run_activation, False),
    "fig1F_stats": ("f_stats", run_activation, True),
}


def main(argv=None):
    run_figure_panels(argv, HERE / "params.yaml", HERE / "outputs", PANELS, WORKFLOWS)


if __name__ == "__main__":
    main()
