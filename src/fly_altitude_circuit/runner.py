"""Shared execution support for explicitly selected paper panels."""

import argparse
import contextlib
import io
from pathlib import Path

from . import plotting

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

from . import config
from .paths import ensure_output_path
from .serialization import write_analysis_report
from .warning_filters import silence_empty_mean


@silence_empty_mean()
def run_figure_panels(argv, params_path, default_output, panels, workflows=None, trace_modes=False):
    """Run selected figure panels and dataset-specific analysis workflows.

    Each panels entry maps an output name to (parameter key, renderer, axes key).
    """
    workflows = workflows or {}
    available = list(panels) + list(workflows)
    parser = argparse.ArgumentParser(description="Generate paper panels and associated analyses.")
    parser.add_argument("--panel", nargs="+", choices=available, help="Default: all panels in this runner.")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--preview", action="store_true", help="Also save PNG previews beside SVGs.")
    parser.set_defaults(no_plot=False)
    statistics = [name for name, spec in workflows.items() if spec[2]]
    if statistics:
        parser.add_argument("--no-plot", action="store_true", help=f"Statistics only; select from {', '.join(statistics)}.")
    parser.set_defaults(trace_mode=None)
    if trace_modes:
        parser.add_argument("--trace-mode", choices=["raw", "diff"], help="Raw by default")
    args = parser.parse_args(argv)
    selected = args.panel or available
    if args.no_plot and any(name not in workflows or not workflows[name][2] for name in selected):
        parser.error(f"--no-plot requires selecting only statistics workflows: {', '.join(statistics)}.")
    params = yaml.safe_load(Path(params_path).read_text())
    screen_selected = any(name in panels for name in selected)
    lines_cfg = config.load("screen_lines") if screen_selected else None
    rng = np.random.default_rng(params["seed"]) if screen_selected else None
    output = ensure_output_path(args.output_dir)
    destinations = {}
    for name in selected:
        if name in workflows:
            continue
        destinations[name] = [ensure_output_path(output / f"{name}.svg")]
        if args.preview:
            destinations[name].append(ensure_output_path(output / f"{name}.png"))
    output.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context():
        plotting.apply_style()
        for name in selected:
            if name in workflows:
                key, execute, _ = workflows[name]
                panel = params["panels"][key]
                if args.trace_mode is not None and "trace_mode" in panel:
                    panel = dict(panel, trace_mode=args.trace_mode)
                execute(name, panel, output, args.preview, not args.no_plot)
                continue
            key, renderer, axes_key = panels[name]
            panel = params["panels"][key]
            kwargs = {"rng": rng} if axes_key == "key_point_axes" else {}
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                fig = renderer(lines_cfg[panel["lines"]], panel, params[axes_key], **kwargs)
            print(console.getvalue(), end="")
            try:
                payload = dict(panel=name, parameters=panel, console=console.getvalue().splitlines())
                report = output / f"{name}.json"
                if write_analysis_report(report, payload):
                    print(f"Saved {report}")
                for destination in destinations[name]:
                    fig.savefig(destination, dpi=config.load("style")["dpi"])
                    print(f"Saved {destination}")
            finally:
                plt.close(fig)



run_screen_panels = run_figure_panels
