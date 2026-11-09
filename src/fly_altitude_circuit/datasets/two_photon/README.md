# Two-photon imaging

Shared implementation for Figures 3/5 and their supplements:

- `io.py`: loading prepared stimulus/behavior and fluorescence dataframes.
- `analysis_and_plotting.py`: calculations and plot construction.
- `execution.py`: execution and exports.
- `rendering.py`: display settings and axis selection.

Input files are configured in [config/datasets.yaml](../../../../config/datasets.yaml).
Shared plotting settings are in [config/two_photon.yaml](../../../../config/two_photon.yaml),
using `layout_mode: CurrBio`. Panel settings live beside each runner.
See [Figure 3](../../../../figures/fig3/README.md) and
[Figure 5](../../../../figures/fig5/README.md) for commands.
