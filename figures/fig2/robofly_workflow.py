"""Figure 2 workflow."""

import argparse
from pathlib import Path

from fly_altitude_circuit import config, plotting
from fly_altitude_circuit.datasets.robofly.io import load_avg_wb_data
from fly_altitude_circuit.paths import ensure_output_path
from fly_altitude_circuit.warning_filters import silence_empty_mean

import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml

from robofly_panels import plot_force_traces


@silence_empty_mean()
def run(argv, params_path, default_output, panels):
    parser = argparse.ArgumentParser(description="Generate robofly paper assets.")
    parser.add_argument("--panel", nargs="+", choices=list(panels))
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--preview", action="store_true", help="Also save PNG previews for force traces.")
    args = parser.parse_args(argv)
    params = yaml.safe_load(Path(params_path).read_text())["panels"]
    selected = args.panel or list(panels)
    output = ensure_output_path(args.output_dir)
    targets = {}
    for name in selected:
        is_overlay = panels[name] == "c"
        suffixes = ["png"] if is_overlay else ["svg", "png"] if args.preview else ["svg"]
        targets[name] = [ensure_output_path(output / f"{name}.{suffix}") for suffix in suffixes]
    output.mkdir(parents=True, exist_ok=True)
    model = load_avg_wb_data()
    with mpl.rc_context():
        plotting.apply_style()
        for name in selected:
            panel = params[panels[name]]
            if panels[name] == "c":
                from fly_altitude_circuit.datasets.robofly.overlay import render_overlay
                scene = render_overlay(model, panel, targets[name][0])
                scene.renWin.Finalize()
            else:
                fig = plot_force_traces(model, panel)
                try:
                    for target in targets[name]:
                        fig.savefig(target, dpi=config.load("style")["dpi"])
                finally:
                    plt.close(fig)
            for target in targets[name]:
                print(f"Saved {target}")
