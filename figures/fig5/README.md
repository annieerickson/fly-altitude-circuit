# Figure 5 — connectomics and two-photon imaging

Run from the repository root with your data location set in `config/paths.yaml`.
Select panels explicitly when switching environments:

```bash
conda activate FAFB-connectome-env
python figures/fig5/run.py --panel fig5A --preview

conda activate MANC-connectome-env
python figures/fig5/run.py --panel fig5D --preview

conda activate altitude-env
python figures/fig5/run.py --panel fig5C_top fig5C_bottom --preview
```

| Selector | Dataset | Environment |
| --- | --- | --- |
| `fig5A` | FAFB cached connectivity | FAFB-connectome-env |
| `fig5C_top` | LPT49 two-photon | altitude-env |
| `fig5C_bottom` | LPT27 two-photon | altitude-env |
| `fig5D` | MANC cached connectivity | MANC-connectome-env |

FAFB/MANC plotting loads prepared tables without credentials or remote queries.
Inputs are under `data_root/connectomics/`; two-photon inputs are under
`data_root/2p/`. Required files are listed in
[config/datasets.yaml](../../config/datasets.yaml).

SVGs go to `figures/fig5/outputs/`; `--preview` adds PNGs. Two-photon panels also
save JSON count reports. `--output-dir PATH` changes the output folder.
Without `--panel`, all four panels run. Use `--help` to list options.
Figure S8 has a [separate runner](supp/README.md).

Panel settings are in [params.yaml](params.yaml); shared settings are in
`config/connectomics.yaml` and `config/two_photon.yaml`. Connectomics figure code
is in `connectomics_workflow.py`, `fafb_panels.py` and `manc_panels.py`.
Loaders/calculations are in
[datasets/connectomics](../../src/fly_altitude_circuit/datasets/connectomics/)
and [datasets/two_photon](../../src/fly_altitude_circuit/datasets/two_photon/).

## Optional connectivity regeneration

Regeneration is separate from plotting and can take a substantial amount of time. It requires
remote-service authentication and writes new tables to the specified directory
outside the input data root.

```bash
conda activate FAFB-connectome-env
python figures/fig5/regenerate_connectomics.py --dataset fafb --output-dir derived/connectomics/FAFB

conda activate MANC-connectome-env
python figures/fig5/regenerate_connectomics.py --dataset manc --output-dir derived/connectomics/MANC
```

FAFB regeneration uses materialization `783` and annotation release `v3.0.0`,
specified independently in `config/connectomics.yaml`. It requires `fafbseg`
and FlyWire authentication. MANC accepts
`NEUPRINT_TOKEN` or `NEUPRINT_TOKEN_FILE` pointing to an external token file.
Do not put credentials in versioned files. Regeneration does not replace the
cached plotting inputs or automatically generate figures.
