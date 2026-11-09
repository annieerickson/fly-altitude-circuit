"""Figure 5 FAFB heatmap."""
from fly_altitude_circuit import config, plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_fafb_heatmap(data_to_plot, *, label, fig_size, axes_titlefont, ticklabelfont, set_color_max=None):
    nt_colormaps = config.load("connectomics")["nt_colormaps"]
    unique_nts = data_to_plot.index.get_level_values('pre_NT').unique()
    colors = np.zeros((len(data_to_plot), 2, 4))
    vmin = data_to_plot.min().min()
    if set_color_max is None:
        vmax = data_to_plot.max().max()
    else:
        vmax = set_color_max
    cbar_kws = {'label': 'Connection Weight', 'use_gridspec': True, 'orientation': 'vertical'}
    fig, ax = plt.subplots(figsize=fig_size)
    for nt in unique_nts:
        mask = data_to_plot.index.get_level_values('pre_NT') == nt
        nt_data = data_to_plot[mask]
        cmap_full = plt.cm.get_cmap(nt_colormaps.get(nt, 'Greys'))
        cmap = lambda x: cmap_full(0.2 + 0.7 * x)
        if vmax > vmin:
            norm_data = (nt_data - vmin) / (vmax - vmin)
        else:
            norm_data = nt_data * 0
        row_indices = np.where(mask)[0]
        for i, row_idx in enumerate(row_indices):
            colors[row_idx, 0] = cmap(norm_data.iloc[i, 0])
            colors[row_idx, 1] = cmap(norm_data.iloc[i, 1])
    for i in range(len(data_to_plot)):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=colors[i, j], edgecolor='white', linewidth=1))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, len(data_to_plot))
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks(np.arange(len(data_to_plot)) + 0.5)
    ax.set_yticklabels([f'{name} ({int(counts)})' for name, nt, counts in data_to_plot.index], fontsize=ticklabelfont)
    ax.set_xticklabels(data_to_plot.columns, fontsize=ticklabelfont)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    fig.suptitle(label, fontsize=axes_titlefont)
    ax.tick_params(axis='both', which='both', length=0)
    for i in range(len(data_to_plot)):
        for j, col in enumerate(['ipsi', 'contra']):
            value = data_to_plot.iloc[i, j]
            ax.text(j + 0.5, i + 0.5, f'{int(np.ceil(value))}', ha='center', va='center', fontsize=ticklabelfont)
    for idx, nt in enumerate(unique_nts):
        mask = data_to_plot.index.get_level_values('pre_NT') == nt
        nt_data = data_to_plot[mask]
        cmap_full = plt.cm.get_cmap(nt_colormaps.get(nt, 'Greys'))
        cmap = LinearSegmentedColormap.from_list(f'{nt}_truncated', [cmap_full(0.2 + 0.7 * i) for i in np.linspace(0, 1, 256)])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        pos = ax.get_position()
        cbar_width = 0.05
        cbar_spacing = 0.1
        cbar_left = pos.x1 + 0.08 + idx * cbar_spacing
        cbar_ax = fig.add_axes([cbar_left, pos.y0 - 0.05, cbar_width, pos.height])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.ax.set_title(nt, pad=2, fontsize=ticklabelfont, rotation=90)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=ticklabelfont)
        if idx != len(unique_nts) - 1:
            cbar.ax.tick_params(left=False, right=False, labelleft=False, labelright=False)
    fig.subplots_adjust(left=0.21, bottom=0.035, right=0.894, top=0.788)
    return fig
