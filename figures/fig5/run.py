"""Run cached connectomics or two-photon Figure 5 panels from this checkout."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'src'))
sys.path.insert(0, str(HERE))
from fly_altitude_circuit.runner import run_figure_panels
import yaml


def execute(name, panel, output, preview=False, plot=True):
    if panel.get('dataset', '').startswith('connectomics_'):
        from connectomics_workflow import execute as workflow
    else:
        from fly_altitude_circuit.datasets.two_photon.execution import execute as workflow
    return workflow(name, panel, output, preview, plot)


def main(argv=None):
    params = yaml.safe_load((HERE / 'params.yaml').read_text())['panels']
    workflows = {name: (name, execute, panel['statistics_only_supported']) for name, panel in params.items()}
    run_figure_panels(argv, HERE / 'params.yaml', HERE / 'outputs', {}, workflows)


if __name__ == '__main__':
    main()
