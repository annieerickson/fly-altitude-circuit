# Figure S8 — two-photon imaging

Run from the repository root with `altitude-env`. Set your data location in
`config/paths.yaml`; prepared inputs are under `data_root/2p/`.

```bash
conda activate altitude-env
python figures/fig5/supp/run.py --panel figS8A figS8B --preview
python figures/fig5/supp/run.py --panel figS8C_top figS8C_bottom --preview
```

| Selector | Output |
| --- | --- |
| `figS8A` | Responses across stimulus speeds |
| `figS8B` | Responses across stimulus widths |
| `figS8C_top` | LPT49 quiescent fluorescence traces |
| `figS8C_bottom` | LPT27 quiescent fluorescence traces |

Without `--panel`, all four outputs run. Use `--help` to list options.
SVGs and JSON count reports go to `figures/fig5/supp/outputs/`; `--preview`
adds PNGs. `--output-dir PATH` changes the output folder.

Settings are in [params.yaml](params.yaml) and
[config/two_photon.yaml](../../../config/two_photon.yaml).
Required files are listed in [config/datasets.yaml](../../../config/datasets.yaml).
Loading, calculations and plotting are shared with Figure 5 in
[datasets/two_photon](../../../src/fly_altitude_circuit/datasets/two_photon/).
