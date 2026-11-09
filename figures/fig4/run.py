"""Silencing and free-flight Figure 4."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from fly_altitude_circuit.runner import run_figure_panels
from silencing_workflow import execute
from free_flight_workflow import execute as execute_free_flight

WORKFLOWS = {
    "fig4EF": ("ef", execute_free_flight, False),
    "fig4EF_stats": ("ef_stats", execute_free_flight, True),
    "fig4B": ("b", execute, False),
    "fig4B_baseline_stats": ("b_baseline", execute, True),
    "fig4B_response_stats": ("b_response", execute, True),
    "fig4C": ("c", execute, False),
    "fig4C_stats": ("c_response", execute, True),
}


def main(argv=None):
    run_figure_panels(argv, HERE / "params.yaml", HERE / "outputs", {}, WORKFLOWS, trace_modes=True)


if __name__ == "__main__":
    main()
