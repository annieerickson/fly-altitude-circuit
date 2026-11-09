# Figure 2 — robotic replay

Run from the repository root with `altitude-env`. Set your data location in
`config/paths.yaml`; inputs are under `data_root/robofly/1070/`.

```bash
conda activate altitude-env
python figures/fig2/run.py --panel fig2B --preview
python figures/fig2/run.py --panel fig2C
```

| Selector | Output |
| --- | --- |
| `fig2B` | Force traces, SVG |
| `fig2C` | Wing-motion and mean-force overlay, PNG |

Figure 2C requires a working graphical VTK/OpenGL session. Without `--panel`,
both panels run, including the 3D renderer. Use `--help` to list options.
Figure S4 has a [separate runner](supp/README.md).

Files go to `figures/fig2/outputs/`; `--preview` also saves `fig2B.png` for `fig2B`.
`--output-dir PATH` changes the output folder. These panels do not write JSON.
Settings, including the 3D camera, are in [params.yaml](params.yaml).
The required MAT result files and matching command text files are listed under
`robofly` in [config/datasets.yaml](../../config/datasets.yaml).

Code: [robofly_workflow.py](robofly_workflow.py) and
[robofly_panels.py](robofly_panels.py). Dataset loading/calculations and 3D
rendering are in [datasets/robofly](../../src/fly_altitude_circuit/datasets/robofly/),
including `overlay.py` and `lollipop_figure.py`.

Code adapted from [Melis et al., 2024](https://github.com/FlyRanch/melis-etal-robofly/tree/69ecc0c3ef1a048daa4173fb1b865071caaeaef3).
