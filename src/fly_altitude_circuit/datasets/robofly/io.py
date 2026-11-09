"""Load the four configured replay pairs"""

import numpy as np

from ...config import load
from ...paths import data_root
from .analysis import RoboAnalysis

RUN_ROLES = ("left_baseline", "right_baseline", "left_stimulation", "right_stimulation")


def replay_paths():
    dataset = load("datasets")["robofly"]
    if tuple(run["role"] for run in dataset["runs"]) != RUN_ROLES:
        raise ValueError("Robofly requires L baseline, R baseline, L stimulation, R stimulation order.")
    root = data_root() / dataset["folder"]
    runs = []
    for run in dataset["runs"]:
        result = root / dataset["results"] / run["result"]
        command = root / dataset["inputs"] / run["input"]
        for path in (result, command):
            if not path.is_file():
                raise FileNotFoundError(f"Required robofly input not found: {path}")
        runs.append((run, result, command))
    return runs


def load_avg_wb_data():
    settings = load("robofly")
    model = RoboAnalysis()
    for name, value in settings["physical"].items():
        setattr(model, name, value)
    model.set_body_cg(np.array(settings["body_cg"]))
    model.set_srf_angle(-(settings["srf_tilt_deg"]/180)*np.pi)
    strk_shift = np.array([[0, -(settings["stroke_shift_deg"]/180)*np.pi]])
    for run, result, command in replay_paths():
        print(f"{run['role']}: {result.name}")
        model.load_mat_file(result.parent, result.name, gain_factor=settings["gain_factor"])
        with command.open("r") as handle:
            command_data = np.loadtxt(handle, delimiter=",")
        frequency = command_data[0:1, 4]
        model.add_freq_and_scaling(freq_fly=frequency, n_wbs=settings["n_wingbeats"])
        model.convert_to_SRF(strk_shift[0, 0], strk_shift[0, 1], wing_side=run["side"])
        model.add_data_to_list_means()
    return model
