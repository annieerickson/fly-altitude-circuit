"""Free-flight workflow"""
import contextlib
import io

from fly_altitude_circuit import config, plotting
from fly_altitude_circuit.datasets.free_flight import analysis
from fly_altitude_circuit.datasets.free_flight.io import load_dataset
from fly_altitude_circuit.paths import ensure_output_path
import matplotlib.pyplot as plt

from free_flight_panels import plot_statistics, plot_traces
from fly_altitude_circuit.serialization import write_analysis_report
from fly_altitude_circuit.warning_filters import silence_empty_mean


def permutation_report(payload):
    """Save only the printed permutation-test lines in the scalar JSON."""
    messages = [line for line in payload['console']
                if line == 'permutation tests' or 'permuted holm corrected p val:' in line]
    return dict(console=messages)


@silence_empty_mean()
def execute(name, panel, output, preview=False, plot=True):
    required = dict(wind=True, optic_flow_grid=True, time='s')
    if any(panel.get(key) != value for key, value in required.items()):
        raise ValueError("Free-flight analysis requires wind=True, optic_flow_grid=True and time='s'.")
    trace = panel['kind'] == 'trace'
    if trace and (panel['long_or_wide'] != 'wide' or not plot):
        raise ValueError('The paper trace retains both original side-by-side axes.')
    destinations = [ensure_output_path(output / f'{name}.json')]
    if plot:
        destinations.append(ensure_output_path(output / f'{name}.svg'))
        if preview:
            destinations.append(ensure_output_path(output / f'{name}.png'))
    experimental, control = load_dataset()
    console = io.StringIO()
    calculate = analysis.calculate_traces if trace else analysis.calculate_statistics
    with contextlib.redirect_stdout(console):
        results = calculate(experimental, control, panel)
    print(console.getvalue(), end='')
    payload = dict(panel=name, parameters=panel, units='meters', results=results, console=console.getvalue().splitlines())
    report = dict(console=payload['console']) if trace else permutation_report(payload)
    if write_analysis_report(destinations[0], report, has_statistics=not trace):
        print(f'Saved {destinations[0]}')
    if plot:
        fig = plot_traces(results, panel) if trace else plot_statistics(results, panel)
        try:
            for path in destinations[1:]:
                fig.savefig(path, dpi=config.load('style')['dpi'])
                print(f'Saved {path}')
        finally:
            plt.close(fig)
    return payload
