# Figure S4 — robotic replay

Run from the repository root with `altitude-env` and your data location set in
`config/paths.yaml`.

```bash
conda activate altitude-env
python figures/fig2/supp/run.py --panel figS4 --preview
```

This generates the force/moment component plots as
`figures/fig2/supp/outputs/figS4.svg`. `--preview` adds PNG;
`--output-dir PATH` changes the output folder. No JSON is written.
VTK/OpenGL is not needed for this panel. Use `--help` to list options.

Settings are in [params.yaml](params.yaml). Inputs are under
`data_root/robofly/1070/`; required files are listed under `robofly` in
[config/datasets.yaml](../../../config/datasets.yaml).
The runner uses [robofly_workflow.py](../robofly_workflow.py) and
[robofly_panels.py](../robofly_panels.py), with loading/calculations in
[datasets/robofly](../../../src/fly_altitude_circuit/datasets/robofly/).
