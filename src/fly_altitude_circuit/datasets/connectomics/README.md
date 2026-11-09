# Connectomics

Shared FAFB/MANC implementation for Figures 5A and 5D:

- `io.py`: loading prepared connectivity tables.
- `fafb.py` and `manc.py`: dataset-specific grouping and ranking.
- `fafb_regeneration.py` and `manc_regeneration.py`: separate remote data retrieval.

Cached plotting does not require remote clients or credentials. Input filenames
are in [config/datasets.yaml](../../../../config/datasets.yaml); remote dataset
settings are in [config/connectomics.yaml](../../../../config/connectomics.yaml).
FAFB regeneration pins production materialization `783` and annotation release
`v3.0.0` independently, including neuron selection and annotation lookups.
See the [Figure 5 guide](../../../../figures/fig5/README.md) for plotting,
regeneration and environment commands.
