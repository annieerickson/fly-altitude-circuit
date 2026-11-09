"""FAFB/MANC heatmaps Figure 5"""
from fly_altitude_circuit import config
from fly_altitude_circuit.paths import ensure_output_path
from fly_altitude_circuit.warning_filters import silence_empty_mean
from fly_altitude_circuit.datasets.connectomics.io import load_cached
import matplotlib.pyplot as plt


@silence_empty_mean()
def execute(name, panel, output, preview=False, plot=True):
    if not plot:
        raise ValueError('Cached connectomics panels do not have a statistics-only mode.')
    destinations = [ensure_output_path(output / f'{name}.svg')]
    if preview:
        destinations.append(ensure_output_path(output / f'{name}.png'))
    if panel['dataset'] == 'connectomics_fafb':
        from fly_altitude_circuit.datasets.connectomics.fafb import prepare_fafb_heatmap as prepare
        from fafb_panels import plot_fafb_heatmap as render
    elif panel['dataset'] == 'connectomics_manc':
        from fly_altitude_circuit.datasets.connectomics.manc import prepare_manc_heatmap as prepare
        from manc_panels import plot_manc_heatmap as render
    else:
        raise ValueError(f'Unsupported connectomics dataset: {panel["dataset"]}')
    table = prepare(load_cached(panel['dataset']), **panel['analysis'])
    style = config.load('style')
    arguments = dict(panel['plot'])
    arguments.setdefault('axes_titlefont', style['font']['axes_title'])
    arguments.setdefault('ticklabelfont', style['font']['ticklabel'])
    before = set(plt.get_fignums())
    try:
        figure = render(table, **arguments)
        for path in destinations:
            figure.savefig(path, dpi=style['dpi'], bbox_inches='tight')
            print(f'Saved {path}')
    finally:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)
    
