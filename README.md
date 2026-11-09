# A neural circuit mediating altitude control in Drosophila: figures and analysis
Analysis code and scripts for generating figure panels and
associated results.

**Paper:** [A neural circuit mediating altitude control in *Drosophila*](https://doi.org/10.1016/j.cub.2026.10.049)

## Data

**Download:** [Datasets on CaltechDATA](https://doi.org/10.22002/j5ncb-76c22)

The datasets are distributed as separate ZIP archives. Create a common `data/`
folder in this checkout or at an external location, then extract and arrange
the datasets so that it contains these subfolders:

```text
data/
├── 2p/
├── connectomics/
├── DNa10_activation/
├── free_flight/
├── optogenetic_screen/
├── robofly/
└── silencing/
```

Input filenames and templates are listed in
[config/datasets.yaml](config/datasets.yaml). Preserve the downloaded dataset
folder structure.

Create your local path configuration:

```bash
cp config/paths.example.yaml config/paths.yaml
```

Set `data_root` in `config/paths.yaml` to the folder containing those subfolders:

```yaml
data_root: /your/local/path/to/data
```

This local configuration is ignored by Git. Relative paths are resolved from the repository root; `~` is supported.

## Environments

| Analyses | Conda environment | Specification |
| --- | --- | --- |
| Screen, activation, robotic replay, imaging, silencing and free flight | altitude-env | [environment.yml](environment.yml) |
| FAFB connectomics | FAFB-connectome-env | [FAFB_environment.yml](environments/FAFB_environment.yml) |
| MANC connectomics | MANC-connectome-env | [MANC_environment.yml](environments/MANC_environment.yml) |

Create the relevant environment from its specification if needed, then activate
it. For example:

```bash
conda env create -f environment.yml
conda activate altitude-env
```

The runners load this checkout's source package directly; no separate package
installation is required.

## Run figures

Run commands from the repository root. Each figure has a `run.py` entry point
and a README listing its panel selectors and options.

```bash
# Generate a selected panel and a PNG preview.
python figures/fig1/run.py --panel fig1D_top --preview

# Generate supplementary panels.
python figures/fig3/supp/run.py --panel figS7_top figS7_bottom --preview

# Run an associated statistical analysis without plotting.
python figures/fig4/run.py --panel fig4EF_stats --no-plot
```

| Figure | Main panels | Supplementary panels |
| --- | --- | --- |
| 1 | [Optogenetic screen and activation](figures/fig1/README.md) | [S1–S2](figures/fig1/supp/README.md) |
| 2 | [Robotic replay](figures/fig2/README.md) | [S4](figures/fig2/supp/README.md) |
| 3 | [Two-photon imaging](figures/fig3/README.md) | [S5–S7](figures/fig3/supp/README.md) |
| 4 | [Silencing and free flight](figures/fig4/README.md) | — |
| 5 | [Connectomics and two-photon imaging](figures/fig5/README.md) | [S8](figures/fig5/supp/README.md) |

Use `--help` to list a runner's options. Without `--panel`, it runs all its
configured outputs. For Figure 5, select panels explicitly in the appropriate
dataset environment. Figure 2C requires a working graphical VTK/OpenGL session.

Generated files go to `outputs/` beside the runner. `--preview` adds PNG copies
of vector plots; `--output-dir PATH` selects another destination. Supported
statistics selectors accept `--no-plot` and save JSON without creating figures.
Reports contain statistics or printed analysis values when available.


## Repository organization

```text
figures/fig1/ … fig5/         Figure runners, plotting helpers and params.yaml
figures/figN/supp/           Related supplementary runners and settings
src/fly_altitude_circuit/    Shared configuration, statistics and plotting utilities
  datasets/                 Dataset-specific loading and calculations
config/                     Dataset filenames and shared analysis/plot settings
environments/               FAFB and MANC environment specifications
```

Panel-specific settings can be found in each figure's `params.yaml`; shared settings
are in `config/`. Dataset implementations can serve more than one figure.

