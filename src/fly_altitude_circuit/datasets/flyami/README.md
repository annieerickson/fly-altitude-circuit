# Optogenetic screen

Loading and calculations for Figure 1D, S1 and S2:

- `io.py`: HDF5 loading using the dataset manifest.
- `metadata.py`: driver labels and display IDs.
- `kinematics.py`: wingbeat reconstruction and per-fly wing-angle summaries.

Input templates are in [config/datasets.yaml](../../../../config/datasets.yaml);
driver ordering is in [config/screen_lines.yaml](../../../../config/screen_lines.yaml).
See the [Figure 1 guide](../../../../figures/fig1/README.md) and
[supplementary guide](../../../../figures/fig1/supp/README.md) for commands.
