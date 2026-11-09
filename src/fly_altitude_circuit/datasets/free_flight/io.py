"""Read the four configured prepared recordings"""
import pandas as pd
from ... import config
from ...paths import data_root

REQUIRED_COLUMNS = {'obj_id_unique', 'time stamp', 'duration', 'z', 'xvel', 'yvel', 'zvel'}


def load_dataset():
    spec = config.load('datasets')['free_flight']
    folder = data_root() / spec['folder']
    groups = []
    for name in ('experimental', 'control'):
        recordings = []
        for filename in spec[name]:
            path = folder / filename
            with path.open('rb') as handle:
                frame = pd.read_csv(handle)
            missing = REQUIRED_COLUMNS.difference(frame.columns)
            if missing:
                raise ValueError(f'{path}: missing columns {sorted(missing)}')
            recordings.append(frame)
        # Concat preserves recording order and repeated index labels.
        groups.append(pd.concat(recordings))
    df_exp, df_control = groups
    df_exp['ground speed'] = (df_exp['xvel']**2 + df_exp['yvel']**2)**0.5
    df_control['ground speed'] = (df_control['xvel']**2 + df_control['yvel']**2)**0.5
    df_exp['xvel_abs'] = abs(df_exp['xvel'])
    df_control['xvel_abs'] = abs(df_control['xvel'])
    df_exp = df_exp.rename(columns={'duration': 'intensity'})
    df_control = df_control.rename(columns={'duration': 'intensity'})
    return df_exp, df_control
