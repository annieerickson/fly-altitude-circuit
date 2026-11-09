"""Read existing combined two-photon dataframes"""
import pandas as pd
from ... import config
from ...paths import data_root


def load_combined(dataset, line):
    spec = config.load('datasets')[dataset]
    if str(line) != spec['line']:
        raise ValueError(f'{dataset} requires source line {spec["line"]}, received {line}')
    folder = data_root() / spec['folder']
    frames = {}
    for kind in ('stimulus', 'fluorescence'):
        path = folder / spec[kind].format(line=line)
        with path.open('rb') as handle:
            # pandas' compatibility unpickler supports the original pandas-1
            # numeric Index classes in the existing pandas-2 environment.
            frames[kind] = pd.read_pickle(handle)
        required = {'fly_id', 'trial_num'}
        required |= {'frameTimes', 'imRateHz'} if kind == 'fluorescence' else {'timeVec'}
        missing = required.difference(frames[kind].columns)
        if missing:
            raise ValueError(f'{path}: missing required columns {sorted(missing)}')
    # Keep original return order, columns, dtypes, indexes and row ordering.
    return frames['fluorescence'], frames['stimulus']


def loom_path():
    spec = config.load('datasets')['two_photon_loom']
    return data_root() / spec['folder'] / spec['filename']
