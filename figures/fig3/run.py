"""Run Figure 3 two-photon panels."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'src'))
from fly_altitude_circuit.runner import run_figure_panels
from fly_altitude_circuit.datasets.two_photon.execution import execute
import yaml


def main(argv=None):
    params = yaml.safe_load((HERE / 'params.yaml').read_text())['panels']
    workflows = {name: (name, execute, panel['statistics_only_supported']) for name, panel in params.items()}
    run_figure_panels(argv, HERE / 'params.yaml', HERE / 'outputs', {}, workflows)


if __name__ == '__main__':
    main()
