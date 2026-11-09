"""Load existing combined silencing dataframes in binary read-only mode."""

import pandas as pd

from ...config import load
from ...paths import data_root

COMMON_COLUMNS = {"driver", "fly_id", "trial_type", "timeVec", "LWF", "RWF", "flying_bool"}


def load_dataframe(dataset):
    spec = load("datasets")[dataset]
    path = data_root() / spec["folder"] / spec["prepared_dataframe"]
    if not path.is_file():
        raise FileNotFoundError(f"Required read-only combined silencing input not found: {path}")
    with path.open("rb") as handle:
        frame = pd.read_pickle(handle)
    required = COMMON_COLUMNS | ({"patV"} if dataset == "silencing_ux" else {"stimVolt", "date", "trial_num"})
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Silencing dataframe is missing columns: {sorted(missing)}")
    return frame
