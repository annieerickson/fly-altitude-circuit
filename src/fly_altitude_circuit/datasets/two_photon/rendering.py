"""Two-photon plotting"""
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle


def apply_axis_settings(fig, settings):
    if not settings:
        return fig
    if 'size_scale' in settings:
        if len(fig.axes) != 1:
            raise ValueError('Axis size scaling requires a single-axis figure.')
        scale = np.asarray(settings['size_scale'], dtype=float)
        if scale.shape != (2,) or not np.all(np.isfinite(scale) & (scale > 0)):
            raise ValueError('Axis size_scale must contain positive width/height factors.')
        original_size = fig.get_size_inches().copy()
        axis = fig.axes[0]
        bounds = np.asarray(axis.get_position().bounds) * np.tile(original_size, 2)
        new_extent = bounds[2:] * scale
        new_size = original_size + new_extent - bounds[2:]
        fig.set_size_inches(*new_size)
        axis.set_position(np.r_[bounds[:2], new_extent] / np.tile(new_size, 2))
    for axis in fig.axes:
        if 'ylim' in settings:
            axis.set_ylim(settings['ylim'])
        if 'left_spine_bounds' in settings:
            axis.spines.left.set_bounds(*settings['left_spine_bounds'])
        if 'ci_zorder' in settings:
            for patch in axis.patches:
                if isinstance(patch, Rectangle):
                    patch.set_zorder(settings['ci_zorder'])
        if 'mean_zorder' in settings:
            for collection in axis.collections:
                if isinstance(collection, LineCollection):
                    collection.set_zorder(settings['mean_zorder'])
    for override in settings.get('subplot_overrides', []):
        selected = [axis for axis in fig.axes
                    if axis.get_subplotspec().rowspan.start == override['row']
                    and axis.get_subplotspec().colspan.start == override['column']]
        if len(selected) != 1:
            raise ValueError('A subplot display override must select exactly one axis.')
        axis = selected[0]
        if 'ylim' in override:
            axis.set_ylim(override['ylim'])
        if 'left_spine_bounds' in override:
            axis.spines.left.set_bounds(*override['left_spine_bounds'])
        for coordinate in ('x', 'y'):
            if f'{coordinate}ticks' not in override:
                continue
            labels = getattr(axis, f'get_{coordinate}ticklabels')()
            font = {'fontproperties': labels[0].get_fontproperties().copy()} if labels else {}
            if f'{coordinate}ticklabel_fontsize' in override:
                font['fontsize'] = override[f'{coordinate}ticklabel_fontsize']
            limits = getattr(axis, f'get_{coordinate}lim')()
            values = override[f'{coordinate}ticks']
            getattr(axis, f'set_{coordinate}ticks')(values, labels=[f'{value:g}' for value in values], **font)
            getattr(axis, f'set_{coordinate}lim')(limits)
        if 'bottom_spine_bounds' in override:
            axis.spines.bottom.set_visible(True)
            axis.spines.bottom.set_bounds(*override['bottom_spine_bounds'])
            axis.spines.bottom.set_linewidth(axis.spines.left.get_linewidth())
        if 'xlabel' in override:
            font = {'fontsize': override['xlabel_fontsize']} if 'xlabel_fontsize' in override else {}
            axis.set_xlabel(override['xlabel'], **font)
        if 'left_spine_position' in override:
            # set_position resets tick artists; retain explicit source fonts.
            fonts = [(tick.label1.get_fontproperties().copy(), tick.label2.get_fontproperties().copy())
                     for tick in axis.yaxis.get_major_ticks()]
            axis.spines.left.set_position(tuple(override['left_spine_position']))
            for tick, (left, right) in zip(axis.yaxis.get_major_ticks(), fonts):
                tick.label1.set_fontproperties(left)
                tick.label2.set_fontproperties(right)
    return fig


def retain_dff_axes(fig, mode, arguments):
    if mode is None:
        return fig
    if mode not in ('dff_row', 'dff_column'):
        raise ValueError(f'Unsupported two-photon axis omission: {mode}')
    fig.canvas.draw()
    original_size = fig.get_size_inches().copy()
    original_axes = list(fig.axes)
    index = arguments.get('dff_row', 0) if mode == 'dff_row' else arguments.get('dff_col', 0)
    retained = [axis for axis in original_axes
                if (axis.get_subplotspec().rowspan.start if mode == 'dff_row'
                    else axis.get_subplotspec().colspan.start) == index]
    if not retained:
        raise ValueError('No dFF axes found in original subplot grid.')
    bounds = {axis: np.array(axis.get_position().bounds) * np.tile(original_size, 2)
              for axis in retained}
    # Keep original outer margins in inches. Remove companion rows/columns and
    # internal gaps; never recompute tight_layout after removing those axes.
    horizontal = mode == 'dff_column'
    coordinate = 0 if horizontal else 1
    extent = 2 if horizontal else 3
    low = min(b[coordinate] for b in bounds.values())
    high = max(b[coordinate] + b[extent] for b in bounds.values())
    source_margin = getattr(fig.subplotpars, 'left' if horizontal else 'bottom') * original_size[coordinate]
    far_margin = (1 - getattr(fig.subplotpars, 'right' if horizontal else 'top')) * original_size[coordinate]
    new_size = original_size.copy()
    new_size[coordinate] = source_margin + high - low + far_margin
    for axis in original_axes:
        if axis not in retained:
            fig.delaxes(axis)
    fig.set_size_inches(*new_size)
    for axis, box in bounds.items():
        box[coordinate] += source_margin - low
        axis.set_position(box / np.tile(new_size, 2))
    return fig
