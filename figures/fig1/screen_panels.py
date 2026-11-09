"""Figure 1D (top, bottom): average wingbeat traces (baseline vs activation) and their difference; and difference of baseline and activation key point per angle"""
from fly_altitude_circuit import config, stats, plotting
from fly_altitude_circuit.datasets.flyami import io
from fly_altitude_circuit.datasets.flyami.metadata import line_names

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from fly_altitude_circuit.datasets.flyami.kinematics import (
    KEY_POINT, key_point_fly_summaries, mean_wingbeat_traces, to_deg)

ANGLES = ["phi", "theta", "eta", "xi"]
ROW = {"phi": 0, "theta": 2, "eta": 4, "xi": 6}      # difference plot is row + 1
LABEL = {"phi": r"$\phi$", "theta": r"$\theta$", "eta": r"$\eta$", "xi": r"$\xi$"}
LABEL_DELTA = {"phi": r"$\Delta\phi$", "theta": r"$\Delta\theta$",
               "eta": r"$\Delta\eta$", "xi": r"$\Delta\xi$"}
GREEK = {"phi": r"\phi", "theta": r"\theta", "eta": r"\eta", "xi": r"\xi"}



def plot_avg_traces_with_changes(lines, panel, axes_cfg):
    """Return a Figure. `panel` and `axes_cfg` come from params.yaml; saving is done by the caller."""
    style = config.load("style")
    font, ln = style["font"], style["line"]
    sign = -1 if panel["baseline_minus_activation"] else 1
    lw = panel.get("line_width", ln["width"])
    label_top, label_bottom = panel["label_top"], panel["label_bottom"]

    fig, axs = plt.subplots(2 * len(ANGLES), len(lines), squeeze=False)

    for col, line in enumerate(lines):
        cell_type, line_id = line_names(line, abbrev=panel["abbrev_labels"])
        label_color = panel["highlight_color"] if line in panel["highlight_lines"] else "black"
        if line in panel["id_only_labels"]:
            title = line_id
        else:
            title = f"{line_id}\n{cell_type}"

        with io.open_line(line, load_wb=not panel["plot_diff_from_avg"]) as (wb, movie):
            for angle in ANGLES:
                row = ROW[angle]
                ax, ax_d = axs[row, col], axs[row + 1, col]

                base, stim = mean_wingbeat_traces(
                    wb, movie, angle, panel["wing_side"], panel["plot_diff_from_avg"])
                t = np.linspace(0, 1, len(base))          # time in units of wingbeat

                ax.plot(t, to_deg(stim), color=panel["trace_color"], linewidth=lw)
                ax.plot(t, to_deg(base), color="black", linewidth=lw)
                ax_d.plot(t, sign * (to_deg(stim) - to_deg(base)),
                          color=panel["change_color"], linewidth=lw)
                ax_d.hlines(0, t[0], t[-1], color="black", linewidth=lw, linestyle="dashed")

                _format_pair(ax, ax_d, angle, row, col, axes_cfg[angle], panel, style)

                if col == 0:
                
                    ax.set_ylabel(LABEL[angle] + "\n (deg)", fontsize=font["axes_title"],
                                rotation=0, va="center", labelpad=panel.get("ylabel_pad"))
                    ax_d.set_ylabel(LABEL_DELTA[angle] + "\n (deg)", fontsize=font["axes_title"],
                                rotation=0, va="center", labelpad=panel.get("ylabel_pad"))

                
                pos = panel["title_position"]            # "top", "bottom" or "none"
                if pos == "top" and row == 0:
                    ax.set_title(title, fontsize=font["driverlabel"], color=label_color,
                                rotation=90, loc="center")
                elif pos == "bottom" and row == 6:
                    ax_d.text(0.5, -0.9, title, fontsize=font["driverlabel"], color=label_color,
                            rotation=90, ha="center", va="center", transform=ax_d.transAxes)

    fig.align_ylabels(axs)
    fig.set_size_inches(*panel["size_in"])
    fig.tight_layout()
    fig.subplots_adjust(**panel["subplots_adjust"])
    if panel.get("top_padding_in", 0):
        # Add blank canvas above the titles without resizing the plotted axes.
        width, height = fig.get_size_inches()
        bottom, top = fig.subplotpars.bottom, fig.subplotpars.top
        new_height = height + panel["top_padding_in"]
        fig.set_size_inches(width, new_height)
        fig.subplots_adjust(bottom=bottom * height / new_height,
                            top=top * height / new_height)
    return fig


def _format_pair(ax, ax_d, angle, row, col, cfg, panel, style):
    """Spines, ticks and limits for one trace axis and the difference axis below it."""
    font, ln = style["font"], style["line"]
    label_top, label_bottom = panel["label_top"], panel["label_bottom"]

    for a in (ax, ax_d):
        for s in ("top", "right", "bottom"):
            a.spines[s].set_visible(False)
        a.spines.left.set_linewidth(ln["spines"])
        a.set_xticks([])
        a.tick_params(axis="both", which="major", labelsize=font["ticklabel"],
                      width=ln["tick_width"], length=ln["tick_length"])
        a.yaxis.set_tick_params(which="major", direction="in")
        a.xaxis.set_tick_params(which="major", direction="in")
        a.tick_params(axis="x", which="major", labelsize=font["driverlabel"],
                      width=ln["tick_width"], length=ln["tick_length"],
                      labeltop=label_top, labelbottom=label_bottom,
                      top=label_top, bottom=label_bottom)

    if col == 0 or panel.get("xlim_all_columns", False):
        ax.set_xlim(-0.25, 1)
        ax_d.set_xlim(-0.25, 1)

    # "wingbeat" axis label: bottom of the last row, or top of the first row
    if label_bottom and row == 6 and col == 0:
        ax_d.set_xlabel("wingbeat", fontsize=font["ticklabel"])
        ax_d.set_xticks([0, 1])
        ax_d.spines["bottom"].set_visible(True)
        ax_d.spines["bottom"].set_linewidth(ln["spines"])
        ax_d.spines.bottom.set_bounds((0, 1))
        ax_d.xaxis.set_label_position("bottom")
    if label_top and row == 0 and col == 0:
        ax.set_xlabel("wingbeat", fontsize=font["ticklabel"], ha="center")
        ax.set_xticks([0, 1])
        ax.spines["top"].set_visible(True)
        ax.spines.top.set_bounds((0, 1))
        ax.xaxis.set_label_position("top")
        ax.xaxis.set_tick_params(which="major", direction="out", pad=2)

    # y limits and ticks come from params.yaml; the spine spans first to last tick
    ylim = cfg.get("ylim_label_top", cfg["ylim"]) if label_top else cfg["ylim"]
    d_ylim = cfg.get("d_ylim_label_bottom", cfg["d_ylim"]) if label_bottom else cfg["d_ylim"]
    for a, lim, ticks in ((ax, ylim, cfg["yticks"]), (ax_d, d_ylim, cfg["d_yticks"])):
        a.set_ylim(lim)
        a.set_yticks(ticks)
        a.spines.left.set_bounds(ticks[0], ticks[-1])

    if col != 0:                        # y axis only on the first column
        for a in (ax, ax_d):
            a.set_yticks([])
            a.spines["left"].set_visible(False)



def _line_label(line, panel):
    cell_type, line_id = line_names(line, abbrev=panel["abbrev_labels"])
    return line_id if line in panel["id_only_labels"] else f"{line_id}\n{cell_type}"


def _ylabel(angle, panel):
    g, point, wc = GREEK[angle], KEY_POINT[angle], panel["wing_combine"]
    return rf"${g}${point}" + panel.get("ylabel_suffix", "")


def plot_key_point_summary(lines, panel, axes_cfg, rng):
    """One axis per angle: bootstrap CI box, grand-mean line, optional per-fly points.
    Returns a Figure; saving is done by the caller."""
    style = config.load("style")
    font, ln = style["font"], style["line"]
    lw = panel.get("line_width", ln["width"])
    angles = panel["angles"]
    sign = -1 if panel["baseline_minus_activation"] else 1

    # compute (each line's files are opened once)
    data = {a: [] for a in angles}
    for line in lines:
        with io.open_line(line) as (_, movie):
            for a in angles:
                data[a].append(key_point_fly_summaries(
                    movie, a, panel["wing_combine"], panel["baseline_subtract"]))

    # plot
    fig, axs = plt.subplots(len(angles), 1, squeeze=False)
    labels = [_line_label(l, panel) for l in lines]
    for row, angle in enumerate(angles):
        ax, cfg = axs[row, 0], axes_cfg[angle]
        for i, (pre, stim, change) in enumerate(data[angle]):
            # Keep the original CI input separate from the displayed reduction.
            ci = stats.create_CI(stim - pre, CI=panel["ci"], iterations=panel["bootstrap_iterations"])
            lo, hi = sorted(to_deg(sign * ci))
            ax.add_patch(Rectangle((i - 0.25, lo), 0.5, hi - lo, facecolor="slategray",
                                   alpha=0.3, edgecolor="none", zorder=1))
            ax.hlines(to_deg(sign * np.mean(change)), i - 0.25, i + 0.25,
                      color=panel["mean_color"], lw=lw, zorder=2)
            if panel["plot_fly_means"]:
                jit = rng.uniform(-panel["jitter"], panel["jitter"], len(change))
                ax.scatter(i + jit, to_deg(sign * change), color="black", s=3,
                           zorder=3, edgecolor="none")
            if row == 0:
                print(f"{labels[i].replace(chr(10), ' ')}: n={len(change)} flies")

        # axes formatting
        ax.set_ylim(cfg["ylim"])
        ax.set_yticks(cfg["yticks"])
        ax.spines.left.set_bounds(cfg["yticks"][0], cfg["yticks"][-1])
        ax.spines.left.set_linewidth(ln["spines"])
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.yaxis.set_tick_params(which="major", direction="in")
        ax.tick_params(axis="y", which="major", labelsize=font["ticklabel"],
                       width=ln["tick_width"], length=ln["tick_length"])
        pad = {} if panel.get("xtick_pad") is None else {"pad": panel["xtick_pad"]}
        ax.tick_params(axis="x", which="major", labelsize=font["driverlabel"],
                       width=ln["tick_width"], length=ln["tick_length"],
                       labeltop=panel["show_line_labels"], labelbottom=False,
                       top=False, bottom=False, rotation=90, **pad)
        if row == 0:
            ax.set_xticks(range(len(lines)))
            ax.set_xticklabels(labels)
        else:
            ax.set_xticks([])
        if panel.get("x_margin") is not None:
            ax.margins(x=panel["x_margin"])
        else:
            ax.set_xlim(-0.5, len(lines) - 0.5)
        ax.set_ylabel(panel.get("ylabel", _ylabel(angle, panel)), fontsize=font["axes_title"], rotation=0,
                      va="center", labelpad=panel.get("ylabel_pad"))

    fig.set_size_inches(*panel["size_in"])
    fig.tight_layout()
    fig.subplots_adjust(**panel["subplots_adjust"])
    fig.align_ylabels(axs[:, 0])
    return fig


