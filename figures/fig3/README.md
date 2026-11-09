# Figure 3 — two-photon imaging

Run from the repository root with `altitude-env`. Set your data location in
`config/paths.yaml`; prepared inputs are under `data_root/2p/`.

```bash
conda activate altitude-env
python figures/fig3/run.py --panel fig3D --preview
python figures/fig3/run.py --panel fig3J fig3K --preview
python figures/fig3/run.py --panel fig3G_stats fig3G_regression --no-plot
```

| Selector | Output |
| --- | --- |
| `fig3D` | Visual-stimulus response traces, SVG |
| `fig3F` | Translational-motion response traces, SVG |
| `fig3G` | Peak fluorescence responses, SVG |
| `fig3G_stats` | Permutation comparison, SVG and statistics JSON |
| `fig3G_regression` | Regression analysis, PNG and statistics JSON |
| `fig3H` | Flight-onset fluorescence, SVG |
| `fig3I` | Upward/static/downward responses, SVG |
| `fig3J` | Visual-stimulus response traces, SVG |
| `fig3K` | Fluorescence tuning curve, SVG |

`--no-plot` is available only for `fig3G_stats` and `fig3G_regression`.
The regression JSON reports mean slope and mean R² with bootstrap intervals.
`fig3G_stats.json` contains only the printed console output.
Without `--panel`, all listed outputs run. Use `--help` to list options.
Related supplements have a [separate runner](supp/README.md).

Files go to `figures/fig3/outputs/`; `--preview` adds PNG previews for vector
outputs. JSON reports contain statistics or printed analysis values/counts when
available. `--output-dir PATH` changes the output folder.

Panel settings are in [params.yaml](params.yaml); shared two-photon plot settings
are in [config/two_photon.yaml](../../config/two_photon.yaml). Required files are listed in
[config/datasets.yaml](../../config/datasets.yaml).
Loading, calculations and plotting are in
[datasets/two_photon](../../src/fly_altitude_circuit/datasets/two_photon/);
`execution.py` coordinates runs and `rendering.py` applies display settings.
