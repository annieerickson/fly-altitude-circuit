"""Two-photon execution"""
import contextlib
import io
import yaml

from ... import config, plotting
from ...paths import ensure_output_path
from ...serialization import write_analysis_report
from ...warning_filters import silence_empty_mean
from . import analysis_and_plotting
from .rendering import apply_axis_settings, retain_dff_axes
import matplotlib.pyplot as plt

# These functions have explicit source sizes and do not consult format.
FIXED_LAYOUT = {
    'plot_peak_dff_by_visual_stim_panel_width_shade_box_CI',
    'plot_peak_dff_by_visual_stim_panel_width_shade_box_CI_stats',
    'plot_peak_dff_by_visual_stim_panel_width_regression',
    'plot_avg_trace_per_flight_starts_per_trial_combined_flies_single_roi',
}


def resolve_panel(panel):
    """Reuse an existing paper panel's parameters"""
    if 'reuse' not in panel:
        return panel
    reference = panel['reuse']
    source = config.CONFIG.parent / reference['params']
    base = yaml.safe_load(source.read_text())['panels'][reference['panel']]
    if 'reuse' in base:
        raise ValueError('Panel reuse must reference a directly configured panel.')
    return {**base, **panel}


@silence_empty_mean()
def execute(name, panel, output, preview=False, plot=True):
    panel = resolve_panel(panel)
    if not plot and not panel['statistics_only_supported']:
        raise ValueError('Statistics-only execution is limited to Figure 3G stats/regression.')
    settings = config.load('two_photon')['settings']
    if panel['function'] not in FIXED_LAYOUT and settings['layout_mode'] is None:
        raise ValueError('Set layout_mode to CurrBio or configure the fallback layout before running this panel.')
    if panel['function'] not in FIXED_LAYOUT and settings['layout_mode'] != 'CurrBio':
        width = settings['column_width_3']
        if not isinstance(width, (int, float)) or width <= 0:
            raise ValueError('The original fallback layout requires a positive column_width_3 before loading data.')
    suffix = panel['export_format']
    destinations = [ensure_output_path(output / f'{name}.json')]
    if plot:
        destinations.append(ensure_output_path(output / f'{name}.{suffix}'))
        if preview and suffix != 'png':
            destinations.append(ensure_output_path(output / f'{name}.png'))
    function = getattr(analysis_and_plotting, panel['function'])
    kwargs = dict(panel['arguments'])
    if panel['statistics_only_supported']:
        kwargs['plot'] = plot
    console = io.StringIO()
    before = set(plt.get_fignums())
    try:
        with contextlib.redirect_stdout(console):
            result = function(panel['datasets'], **kwargs)
        print(console.getvalue(), end='')
        figure = result['figure']
        if plot:
            retain_dff_axes(figure, panel['retain'], panel['arguments'])
            apply_axis_settings(figure, panel.get('axes'))
            plotting.add_canvas_padding(figure, panel.get('canvas_padding_in'))
        payload = dict(panel=name, parameters=panel, results=result['results'], console=console.getvalue().splitlines())
        report = dict(console=payload['console']) if name == 'fig3G_stats' else payload
        if write_analysis_report(destinations[0], report, has_statistics=bool(result['results'])):
            print(f'Saved {destinations[0]}')
        if plot:
            for path in destinations[1:]:
                figure.savefig(path, dpi=config.load('style')['dpi'], transparent=panel['transparent'])
                print(f'Saved {path}')
        return payload
    finally:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)
