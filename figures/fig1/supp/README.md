# Figures S1 and S2 — optogenetic screen

Run from the repository root with `altitude-env` and your data location set in
`config/paths.yaml`.

```bash
conda activate altitude-env
python figures/fig1/supp/run.py --panel figS1_top figS1_bottom --preview
python figures/fig1/supp/run.py --panel figS2 --preview
```

| Selector | Output |
| --- | --- |
| `figS1_top` | Left-wing traces |
| `figS1_bottom` | Right-wing traces |
| `figS2` | Bilateral wing-angle summaries |

Without `--panel`, all three outputs run. Use `--help` to list options.
SVGs go to `figures/fig1/supp/outputs/`; `--preview` adds PNGs.
S2 also saves a JSON count report. Use `--output-dir PATH` to choose another folder.

Settings are in [params.yaml](params.yaml); inputs are under
`data_root/optogenetic_screen/`, with file templates in
[config/datasets.yaml](../../../config/datasets.yaml).
Plotting uses [screen_panels.py](../screen_panels.py); loading/calculations are
in [datasets/flyami](../../../src/fly_altitude_circuit/datasets/flyami/).
