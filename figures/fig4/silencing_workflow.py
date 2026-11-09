"""Silencing workflows; figures are optional for statistics."""


from fly_altitude_circuit import config, plotting
from fly_altitude_circuit.datasets.silencing import analysis
from fly_altitude_circuit.datasets.silencing.io import load_dataframe
from fly_altitude_circuit.paths import ensure_output_path
from fly_altitude_circuit.serialization import write_analysis_report
from fly_altitude_circuit.warning_filters import silence_empty_mean

import matplotlib.pyplot as plt
import numpy as np

from silencing_panels import plot_statistics, plot_traces

EXTRACTORS = {"ux_trace": analysis.extract_ux_traces, "ux_baseline": analysis.extract_ux_baseline,
              "ux_response": analysis.extract_ux_response, "gtacr_trace": analysis.extract_gtacr_traces,
              "gtacr_response": analysis.extract_gtacr_response}



@silence_empty_mean()
def execute(name, panel, output, preview=False, plot=True):
    analysis.validate_paper_settings(panel)
    trace = panel["kind"].endswith("trace")
    if trace and not plot:
        raise ValueError("Statistics-only mode requires a scalar statistics workflow.")
    basename = name if not trace or panel["trace_mode"] == "raw" else f"{name}_diff"
    destinations = [ensure_output_path(output / f"{basename}.json")]
    if plot:
        destinations.append(ensure_output_path(output / f"{basename}.svg"))
        if preview:
            destinations.append(ensure_output_path(output / f"{basename}.png"))
    frame = load_dataframe(panel["dataset"])
    extracted = EXTRACTORS[panel["kind"]](frame, panel)
    del frame
    results = analysis.calculate_results(extracted, panel)
    lines = []
    for group in extracted["groups"]:
        lines.append(f"driver: {group['driver']} fly n: {group['fly_n']} pattern presentations: {group['pattern_n']}")
    if "effect" in results:
        effect = results["effect"]
        if effect["relative_unit"] == "percent":
            lines.append(f"percent increase in baseline WSA: {np.round(effect['relative'], 2)}; CI {np.round(effect['relative_ci'], 2)}")
        else:
            lines.append(f"reduction of behavior: {effect['relative']}; CI {effect['relative_ci']}")
    if "pvalue" in results:
        lines += ["permutation tests", f"permutation p val: {results['pvalue']:.3e}"]
    print("\n".join(lines))
    counts = [{k: group[k] for k in ("driver", "fly_ids", "fly_n", "pattern_n", "presentations")} for group in extracted["groups"]]
    payload = dict(panel=name, parameters=panel, units="degrees", counts=counts,
                   results=results, console=lines)
    if write_analysis_report(destinations[0], dict(console=lines), has_statistics=not trace):
        print(f"Saved {destinations[0]}")
    if plot:
        fig = plot_traces(results, panel) if trace else plot_statistics(results, panel)
        try:
            for destination in destinations[1:]:
                fig.savefig(destination, dpi=config.load("style")["dpi"])
                print(f"Saved {destination}")
        finally:
            plt.close(fig)
    return payload
