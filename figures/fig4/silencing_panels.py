"""Silencing plots Figure 4."""

from fly_altitude_circuit import config, plotting

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def _ticks(axis, explicit_labels=False):
    style = config.load("style")
    line, font = style["line"], style["font"]
    for coordinate in (axis.xaxis, axis.yaxis):
        coordinate.set_tick_params(which="major", direction="in", width=line["tick_width"],
                                   length=line["tick_length"], labelsize=font["ticklabel"])
    for spine in ("left", "bottom"):
        axis.spines[spine].set_linewidth(line["spines"])
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if explicit_labels:
        axis.set_xticklabels(axis.get_xticks(), fontsize=font["ticklabel"])
        axis.set_yticklabels(axis.get_yticks(), fontsize=font["ticklabel"])


def trace_axis_inches(panel):
    """Original first-axis bounds in inches, independent of selected content."""
    width, height = panel["source_size_in"]
    layout = panel["source_layout"]
    axis_width = (layout["right"] - layout["left"]) / (2 + layout["wspace"])
    return (layout["left"] * width, layout["bottom"] * height,
            axis_width * width, (layout["top"] - layout["bottom"]) * height)


def plot_traces(results, panel):
    mode = panel["trace_mode"]
    if mode not in ("raw", "diff"):
        raise ValueError("Silencing trace mode must be raw or diff.")
    style = config.load("style")
    visual, font = style["silencing"], style["font"]
    source_width, height = panel["source_size_in"]
    width = source_width / 2
    left, bottom, axis_width, axis_height = trace_axis_inches(panel)
    # Retain one original subplot's dimensions, fonts and aspect. Both selected
    # modes use its original left margin to give their labels standalone space.
    fig = plt.figure(figsize=(width, height))
    axis = fig.add_axes([left / width, bottom / height, axis_width / width, axis_height / height])
    ux = panel["kind"] == "ux_trace"
    for group in results["groups"]:
        colors = visual["trace_colors"]
        color = colors.get(group["driver"], colors["default"]) if not ux else colors["S000"] if group["driver"] == "S000" else visual["ux_experiment_color"]
        if ux and panel["include_visual_patch"]:
            low = -30 if mode == "raw" else -55
            axis.fill_between([0, panel["stim_duration"]], low, 160,
                              color=visual["visual_color"], alpha=visual["visual_alpha"],
                              linewidth=visual["patch_line_width"])
        time, mean, interval = group["time"], group[mode], group[f"{mode}_ci"]
        selection = slice(None) if ux else slice(None, -2)
        axis.plot(time[selection], mean[selection], color=color, linewidth=panel["line_width"])
        axis.fill_between(time[selection], interval[selection, 0], interval[selection, 1],
                          color=color, alpha=panel["alpha_CI"], edgecolor="none")
    if not ux:
        upper = 160 if mode == "raw" else 130
        axis.fill_between([0, panel["stim_duration"]], -30, upper,
                          color=visual["illumination_color"], alpha=visual["illumination_alpha"],
                          linewidth=visual["patch_line_width"])
    axes = panel["axes"]
    setting = axes[mode]
    axis.set_xlim(*axes["xlim"])
    axis.set_xticks(axes["xticks"])
    axis.set_ylim(*setting["ylim"])
    axis.set_yticks(setting["yticks"])
    axis.spines.left.set_bounds(tuple(setting["spine_bounds"]))
    axis.spines.bottom.set_bounds((0, panel["stim_duration"]))
    label = (r"$\Delta$" if mode == "diff" else "") + "L+R WSA (deg)"
    axis.set_ylabel(label, rotation=90, ha="center", fontsize=font["axes_title"], labelpad=0)
    axis.set_xlabel("time (s)", fontsize=font["axes_title"], labelpad=-1)
    _ticks(axis, explicit_labels=True)
    return fig


def plot_statistics(results, panel):
    style = config.load("style")
    visual, font, line = style["silencing"], style["font"], style["line"]
    fig, axis = plt.subplots(1, 1)
    for index, group in enumerate(results["groups"]):
        bounds = group["ci"]
        half_width = visual["interval_width"] / 2
        axis.add_patch(Rectangle((index - half_width, bounds[0]), width=2*half_width,
            height=bounds[1]-bounds[0], facecolor=visual["interval_color"],
            alpha=visual["interval_alpha"], edgecolor="none", zorder=1))
        axis.hlines(group["mean"], index-half_width, index+half_width,
                    color=visual["mean_color"], lw=line["width"], zorder=2)
        values = group["values"]
        axis.scatter(np.full_like(values, index) + group["jitter"], values,
                     color=visual["point_color"], alpha=1.0, s=visual["point_size"],
                     zorder=3, edgecolor="none", marker="o")
    axes = panel["axes"]
    axis.set_ylim(*axes["ylim"])
    axis.set_yticks(axes["yticks"])
    axis.spines.left.set_bounds(tuple(axes["spine_bounds"]))
    axis.set_xlim(-0.6, 1.6)
    axis.set_xticks([0, 1], axes["labels"], ha="right")
    axis.spines.bottom.set_bounds((0, 1))
    axis.set_ylabel(axes["ylabel"], rotation=90, ha="center", fontsize=font["axes_title"], labelpad=axes["ylabel_pad"])
    _ticks(axis)
    axis.xaxis.set_tick_params(which="major", rotation=axes["rotation"])
    if panel["kind"] == "ux_response":
        axis.set_yticklabels(axis.get_yticks(), fontsize=font["ticklabel"])
    fig.set_size_inches(*panel["size_in"])
    fig.subplots_adjust(**panel["layout"])
    plotting.add_canvas_padding(fig, panel.get("canvas_padding_in"))
    return fig
