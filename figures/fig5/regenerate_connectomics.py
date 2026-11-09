"""Regenerate connectivity data.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['fafb', 'manc'])
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='Writable derived-table directory outside the read-only data root.')
    args = parser.parse_args(argv)
    from fly_altitude_circuit import config
    from fly_altitude_circuit.datasets.connectomics.regeneration_paths import output_directory
    # Validate before remote imports, credential reads or service requests.
    destination = output_directory(args.output_dir)
    settings = config.load('connectomics')[args.dataset]
    if args.dataset == 'fafb':
        from fly_altitude_circuit.datasets.connectomics.fafb_regeneration import (
            get_all_input_ds, generate_df_with_all_inputs_and_types_with_NT)
        if settings['dataset'] != 'production':
            raise ValueError('The preserved FAFB workflow requires the production dataset.')
        connections = get_all_input_ds(
            cell_type=settings['cell_type'], downstream_root_ids=None,
            upstream=True, downstream=False, save_df=True, save_dir=destination,
            materialization=settings['materialization'],
            annotation_version=settings['annotation_version'])
        generate_df_with_all_inputs_and_types_with_NT(
            connections, synapse_thresh=True, synapse_thresh_val=settings['synapse_thresh_val'],
            save_df=True, save_dir=destination, cell_type_name=settings['cell_type'],
            materialization=settings['materialization'],
            annotation_version=settings['annotation_version'])
    else:
        token = os.environ.get(settings['token_env'])
        token_file = os.environ.get(settings['token_file_env'])
        if not token and token_file:
            token = Path(token_file).expanduser().read_text().strip()
        if not token:
            raise ValueError('Set NEUPRINT_TOKEN or NEUPRINT_TOKEN_FILE for explicit MANC regeneration.')
        from neuprint import Client
        from fly_altitude_circuit.datasets.connectomics.manc_regeneration import retrieve_downstream_connectivity_all_cells
        client = Client(server=settings['server'], dataset=settings['dataset'], token=token)
        retrieve_downstream_connectivity_all_cells(
            cell_type=settings['cell_type'], synapse_thresh_val=settings['synapse_thresh_val'],
            write_to_csv=True, save_dir=destination, client=client)
    print(f'Regenerated {args.dataset.upper()} tables saved under {destination}. Cached data was not changed.')


if __name__ == '__main__':
    main()
