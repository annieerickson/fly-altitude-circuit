"""Figure 1F and associated activation statistics: rendering only."""

from fly_altitude_circuit import config, plotting

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def _format_axes(ax, panel, labels=None):
    style = config.load("style")
    font, line = style["font"], style["line"]
    axes = panel["axes"]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(*axes["ylim"])
    ax.set_yticks(axes["yticks"])
    ax.spines.left.set_bounds(tuple(axes["left_spine_bounds"]))
    ax.set_xticks(axes["xticks"])
    if "xlim" in axes:
        ax.set_xlim(*axes["xlim"])
        ax.spines.bottom.set_bounds(tuple(axes["bottom_spine_bounds"]))
    ax.set_ylabel(axes["ylabel"], rotation=90, ha="center", fontsize=font["axes_title"], labelpad=axes["ylabel_pad"])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_tick_params(which="major", direction="in", width=line["tick_width"], length=line["tick_length"])
    ax.spines.left.set_linewidth(line["spines"])
    ax.spines.bottom.set_linewidth(line["spines"])
    ax.set_xticklabels(ax.get_xticks() if labels is None else labels, fontsize=font["ticklabel"])
    ax.set_yticklabels(ax.get_yticks(), fontsize=font["ticklabel"])
    if "xlabel" in axes:
        ax.set_xlabel(axes["xlabel"], fontsize=font["axes_title"])


def _layout(fig, panel):
    fig.set_size_inches(*panel["size_in"])
    fig.tight_layout()
    fig.subplots_adjust(**panel["subplots_adjust"])


def plot_activation_traces(results, panel):
    style = config.load("style")["activation"]
    fig, ax = plt.subplots(1, 1)
    for group in results:
        colors = style["trace_colors"]
        color = colors.get(group["driver"], colors["default"])
        time, mean, interval = group["time"], group["mean"], group["ci"]
        ax.plot(time[:-2], mean[:-2], color=color, linewidth=panel["line_width"])
        if interval is not None:
            ax.fill_between(time[:-2], interval[:-2, 0], interval[:-2, 1],
                            color=color, alpha=panel["alpha_CI"], edgecolor="none")
    _format_axes(ax, panel)
    ax.fill_between([0, panel["stim_duration"]], *panel["axes"]["ylim"],
                    color=style["stimulus_color"], alpha=panel["alpha_patch"],
                    linewidth=style["stimulus_line_width"])
    _layout(fig, panel)
    return fig


def plot_activation_responses(results, panel):
    style = config.load("style")["activation"]
    fig, ax = plt.subplots(1, 1)
    for index, group in enumerate(results["groups"]):
        values, bounds = group["values"], group["ci"]
        ax.scatter(np.full_like(values, index) + group["jitter"], values,
                   color=style["point_color"], alpha=1.0, s=style["point_size"],
                   zorder=3, edgecolor="none")
        width = style["interval_width"]
        ax.add_patch(Rectangle((index - width/2, bounds[0]), width=width,
                               height=bounds[1] - bounds[0],
                               facecolor=style["interval_color"], alpha=style["interval_alpha"],
                               edgecolor="none", zorder=1))
        ax.hlines(group["mean"], index - width/2, index + width/2,
                  color=style["mean_color"], lw=panel["line_width"], zorder=2)
    _format_axes(ax, panel, [group["label"] for group in results["groups"]])
    _layout(fig, panel)
    return fig
