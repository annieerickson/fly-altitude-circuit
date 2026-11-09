"""Read prepared activation data."""

import pandas as pd

from ...config import load
from ...paths import data_root

REQUIRED_COLUMNS = {"trial_type", "driver", "fly_id", "timeVec", "stimVolt",
                    "flying_bool", "LWF", "RWF"}


def load_dataframe():
    settings = load("datasets")["DNa10_activation"]
    path = data_root() / settings["folder"] / settings["prepared_dataframe"]
    if not path.is_file():
        raise FileNotFoundError(f"Required activation input not found: {path}")
    with path.open("rb") as handle:
        frame = pd.read_pickle(handle)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Activation dataframe is missing columns: {sorted(missing)}")
    return frame
