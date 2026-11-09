"""Figure 2."""

from fly_altitude_circuit import config, plotting
from fly_altitude_circuit.datasets.robofly.analysis import bilateral_force_traces

import matplotlib.pyplot as plt


def plot_force_traces(model, panel):
    style = config.load("style")
    paper, font, line = style["robofly"], style["font"], style["line"]
    components = panel["components"]
    fig, axes = plt.subplots(1, len(components), squeeze=False)
    axes = axes[0]
    for position, (axis, component) in enumerate(zip(axes, components)):
        for index, result in enumerate(bilateral_force_traces(model, component)):
            color = paper["baseline_color"] if index == 0 else paper["stimulation_color"]
            alpha = paper["baseline_mean_alpha"] if index == 0 else paper["stimulation_mean_alpha"]
            mean_style = paper["baseline_mean_style"] if index == 0 else paper["stimulation_mean_style"]
            time = result["time"]
            axis.plot(time, result["trace"], color=color, linewidth=line["width"])
            axis.hlines(result["mean"], time[0], time[-1], linestyle=mean_style,
                        color=color, alpha=alpha, linewidth=line["width"])
        if len(components) == 1:
            axis.set_ylabel(panel["ylabel"], fontsize=font["axes_title"], rotation=90,
                            labelpad=paper["ylabel_pad"])
            axis.set_title(panel["title"], fontsize=font["axes_title"])
            ylim, ticks, bounds = panel["ylim"], panel["yticks"], panel["spine_bounds"]
        else:
            axis.set_title(panel["component_titles"][position], fontsize=font["axes_title"],
                           pad=paper["component_title_pad"])
            ylim, ticks, bounds = panel["ylims"][position], panel["yticks"][position], panel["spine_bounds"][position]
            axis.set_xlim(*panel["xlim"])
        axis.set_ylim(*ylim)
        axis.set_yticks(ticks)
        axis.spines.left.set_bounds(tuple(bounds))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["bottom"].set_visible(True)
        axis.spines.bottom.set_bounds((0, 1))
        axis.set_xticks([0, 1])
        axis.set_xlabel(panel.get("xlabel", paper["xlabel"]), fontsize=font["axes_title"],
                        labelpad=panel.get("xlabel_pad", paper["xlabel_pad"]))
        for coordinate in (axis.xaxis, axis.yaxis):
            coordinate.set_tick_params(which="major", direction="in", width=line["tick_width"], length=line["tick_length"])
        axis.spines.left.set_linewidth(line["spines"])
        axis.spines.bottom.set_linewidth(line["spines"])
        axis.set_xticklabels(axis.get_xticks(), fontsize=font["ticklabel"])
        axis.set_yticklabels(axis.get_yticks(), fontsize=font["ticklabel"])
    if len(components) > 1:
        fig.suptitle(panel["title"], fontsize=font["axes_title"])
        fig.align_ylabels(axes[:])
    fig.set_size_inches(*panel["size_in"])
    fig.tight_layout()
    fig.subplots_adjust(**panel["subplots_adjust"])
    if panel.get("bottom_padding_in", 0):
        # Add canvas below the plot, preserving physical axis dimensions.
        width, height = fig.get_size_inches()
        bottom, top = fig.subplotpars.bottom, fig.subplotpars.top
        padding = panel["bottom_padding_in"]
        new_height = height + padding
        fig.set_size_inches(width, new_height)
        fig.subplots_adjust(bottom=(bottom * height + padding) / new_height,
                            top=(top * height + padding) / new_height)
    return fig
