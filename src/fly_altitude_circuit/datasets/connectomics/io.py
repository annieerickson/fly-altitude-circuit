"""Load prepared connectomics tables without remote access."""
import pandas as pd
from ... import config
from ...paths import data_root


REQUIRED = {
    'connectomics_fafb': {'pre', 'post', 'weight', 'pre_cell_type', 'pre_side', 'pre_super_class', 'pre_NT'},
    'connectomics_manc': {'bodyId_pre', 'bodyId_post', 'type_post', 'pre_side', 'post_side', 'side', 'weight'},
}


def input_path(dataset):
    spec = config.load('datasets')[dataset]
    return data_root() / spec['folder'] / spec['prepared_dataframe']


def load_cached(dataset):
    if dataset not in REQUIRED:
        raise ValueError(f'Unsupported connectomics dataset: {dataset}')
    path = input_path(dataset)
    if not path.is_file():
        raise FileNotFoundError(f'Required cached connectomics input not found: {path}. Check config/paths.yaml; regeneration is never automatic.')
    if dataset == 'connectomics_fafb':
        with path.open('rb') as handle:
            frame = pd.read_pickle(handle)
    elif dataset == 'connectomics_manc':
        frame = pd.read_csv(path)
    missing = REQUIRED[dataset].difference(frame.columns)
    if missing:
        raise ValueError(f'{path}: missing required columns {sorted(missing)}')
    return frame
