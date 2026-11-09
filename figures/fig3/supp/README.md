# Figures S5–S7 — two-photon imaging

Run from the repository root with `altitude-env`. Set your data location in
`config/paths.yaml`; prepared inputs are under `data_root/2p/`.

```bash
conda activate altitude-env
python figures/fig3/supp/run.py --panel figS5C figS5D figS5E figS5F --preview
python figures/fig3/supp/run.py --panel figS6 --preview
python figures/fig3/supp/run.py --panel figS7_top figS7_bottom --preview
```

| Selector | Output |
| --- | --- |
| `figS5C` | Trial traces, SVG |
| `figS5D` | Descending-branch flight-onset traces, SVG |
| `figS5E` | Left/right fluorescence correlation, PNG |
| `figS5F` | Descending-branch visual responses, SVG |
| `figS6` | Visual-stimulus response traces, SVG |
| `figS7_top` | Fluorescence traces using Figure 3D parameters, SVG |
| `figS7_bottom` | Quiescent fluorescence traces, SVG |

Without `--panel`, all listed outputs run. Use `--help` to list options.
Files go to `figures/fig3/supp/outputs/`; `--preview` adds PNG previews for SVGs.
JSON records printed analysis values/counts when available.
`--output-dir PATH` changes the output folder.

Settings are in [params.yaml](params.yaml) and
[config/two_photon.yaml](../../../config/two_photon.yaml).
Required files are listed in [config/datasets.yaml](../../../config/datasets.yaml).
Loading, calculations and plotting are shared with Figure 3 in
[datasets/two_photon](../../../src/fly_altitude_circuit/datasets/two_photon/).
