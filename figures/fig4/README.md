# Figure 4 — silencing and free flight

Run from the repository root with `altitude-env`. Set your data location in
`config/paths.yaml`; inputs are under `data_root/silencing/` and
`data_root/free_flight/`.

```bash
conda activate altitude-env
python figures/fig4/run.py --panel fig4B fig4C --preview
python figures/fig4/run.py --panel fig4B fig4C --trace-mode diff --preview
python figures/fig4/run.py --panel fig4EF --preview
python figures/fig4/run.py --panel fig4B_baseline_stats fig4B_response_stats fig4C_stats fig4EF_stats --no-plot
```

| Selector | Output |
| --- | --- |
| `fig4B` | Reaper-hid silencing traces |
| `fig4B_baseline_stats` | Baseline comparison |
| `fig4B_response_stats` | Response comparison |
| `fig4C` | GtACR silencing traces |
| `fig4C_stats` | Response comparison |
| `fig4EF` | Free-flight altitude traces: light on and light off |
| `fig4EF_stats` | Free-flight altitude-change comparisons |

Silencing traces use raw values by default; `--trace-mode diff` selects
baseline-subtracted traces and adds `_diff` to their filenames. This option
applies only to `fig4B` and `fig4C`. Statistics selectors support `--no-plot`;
omit it to also generate summary plots. Without `--panel`, all seven outputs
run. Use `--help` to list options.

SVGs and JSON reports go to `figures/fig4/outputs/`; `--preview` adds PNGs.
JSON reports contain only printed console output.
`fig4EF_stats.json` contains the printed permutation-test results.
`--output-dir PATH` changes the output folder.

Settings are in [params.yaml](params.yaml); input filenames are listed in
[config/datasets.yaml](../../config/datasets.yaml).
Figure code is in `silencing_workflow.py`, `silencing_panels.py`,
`free_flight_workflow.py` and `free_flight_panels.py`. Loaders/calculations are in
[datasets/silencing](../../src/fly_altitude_circuit/datasets/silencing/) and
[datasets/free_flight](../../src/fly_altitude_circuit/datasets/free_flight/).
