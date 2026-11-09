"""Activation workflow: analysis, optional plot, outputs."""

from fly_altitude_circuit import config, plotting
from fly_altitude_circuit.datasets.activation.io import load_dataframe
from fly_altitude_circuit.datasets.activation.analysis import (
    calculate_response_results, calculate_trace_results, extract_response_data,
    extract_trace_data, extraction_settings,
)
from fly_altitude_circuit.paths import ensure_output_path
from fly_altitude_circuit.serialization import write_analysis_report
from fly_altitude_circuit.warning_filters import silence_empty_mean

import matplotlib.pyplot as plt

from activation_panels import plot_activation_responses, plot_activation_traces


@silence_empty_mean()
def run_activation(name, panel, output, preview=False, plot=True):
    summary = name == "fig1F_stats"
    if not summary and not plot:
        raise ValueError("Statistics-only mode is available for fig1F_stats.")
    destinations = [ensure_output_path(output / f"{name}.json")]
    if plot:
        destinations.append(ensure_output_path(output / f"{name}.svg"))
        if preview:
            destinations.append(ensure_output_path(output / f"{name}.png"))
    frame = load_dataframe()
    extractor = extract_response_data if summary else extract_trace_data
    groups = extractor(frame, **extraction_settings(panel))
    del frame
    results = calculate_response_results(groups, panel) if summary else calculate_trace_results(groups, panel)
    lines = []
    if not summary:
        for group in groups:
            line = f"driver: {group['driver']} fly n: {group['fly_n']}"
            lines.append(line)
    if summary:
        lines.append("permutation tests")
        for (x, y), pvalue in zip(results["comparisons"], results["corrected_pvalues"]):
            first, second = results["groups"][x]["label"], results["groups"][y]["label"]
            lines.append(f" {first} x {second} permutation holm corrected p val: {pvalue:.3e}")
    print("\n".join(lines))
    payload = dict(panel=name, parameters=panel,
                   units="degrees", counts=[{key: g[key] for key in ("driver", "fly_ids", "fly_n", "pattern_n", "presentations")} for g in groups],
                   results=results, console=lines)
    reported = write_analysis_report(destinations[0], dict(console=lines), has_statistics=summary)
    if reported and not summary:
        print(f"Saved {destinations[0]}")
    if plot:
        fig = plot_activation_responses(results, panel) if summary else plot_activation_traces(results, panel)
        try:
            for destination in destinations[1:]:
                fig.savefig(destination, dpi=config.load("style")["dpi"])
                if not summary:
                    print(f"Saved {destination}")
        finally:
            plt.close(fig)
    return payload
