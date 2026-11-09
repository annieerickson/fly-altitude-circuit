# Figure 1 — optogenetic screen and activation

Run from the repository root with `altitude-env`. Set your data location in
`config/paths.yaml` before running.

```bash
conda activate altitude-env
python figures/fig1/run.py --panel fig1D_top fig1D_bottom --preview
python figures/fig1/run.py --panel fig1F --preview
python figures/fig1/run.py --panel fig1F_stats --no-plot
```

| Selector | Output |
| --- | --- |
| `fig1D_top` | Baseline/activation wingbeat traces and changes |
| `fig1D_bottom` | Bilateral wing-angle summary |
| `fig1F` | Activation response traces |
| `fig1F_stats` | Activation response summary and permutation tests |

Omit `--no-plot` to also plot `fig1F_stats`. Without `--panel`, all listed
outputs run. Use `--help` to list options. Figures S1/S2 have a
[separate runner](supp/README.md).

Outputs are SVGs in `figures/fig1/outputs/`; `--preview` adds PNGs.
JSON reports contain statistics or printed analysis values/counts when available.
`fig1F.json` contains only the printed driver/fly counts; `fig1F_stats.json`
contains only the printed Holm-corrected permutation-test results.
`--output-dir PATH` changes the output folder. Statistics-only mode saves JSON
without creating a plot.

Settings are in [params.yaml](params.yaml). Screen inputs come from
`data_root/optogenetic_screen/`; activation uses `data_root/DNa10_activation/df.pkl`.
Input filenames and templates are listed in
[config/datasets.yaml](../../config/datasets.yaml).

Code: [screen_panels.py](screen_panels.py),
[activation_workflow.py](activation_workflow.py), and
[activation_panels.py](activation_panels.py). Loaders/calculations are under
[datasets/flyami](../../src/fly_altitude_circuit/datasets/flyami/) and
[datasets/activation](../../src/fly_altitude_circuit/datasets/activation/).
