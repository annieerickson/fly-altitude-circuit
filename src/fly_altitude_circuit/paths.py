"""Resolve input and output paths without writing into the read-only data tree."""

import os
from pathlib import Path

import yaml

from .config import CONFIG

def _resolve_root(value):
    root = Path(value).expanduser()
    return (CONFIG.parent / root).resolve() if not root.is_absolute() else root.resolve()


def data_root():
    """Read the configured input location; never create or modify it."""
    settings = yaml.safe_load((CONFIG / "paths.yaml").read_text())
    return _resolve_root(settings["data_root"])


def protected_data_roots():
    """Protect the active input tree and any additional locally listed roots."""
    local_config = CONFIG / "paths.yaml"
    if not local_config.exists():
        return []
    settings = yaml.safe_load(local_config.read_text())
    return [data_root(), *(_resolve_root(value) for value in settings.get("protected_data_roots", []))]


def ensure_output_path(path):
    """Reject paths inside either the configured or protected input tree.

    Resolve symlinks before checking. This function does not create a directory
    or file, so callers can also validate destinations before mkdir/savefig.
    """
    output = Path(path).expanduser().resolve()
    for root in protected_data_roots():
        root = root.resolve()
        if output.is_relative_to(root):
            raise ValueError(f"Output destination is inside the read-only data root: {output}")
    return output


def configure_matplotlib_cache(default=None):
    """Validate the cache location before importing Matplotlib."""
    candidate = os.environ.get("MPLCONFIGDIR") or default or CONFIG.parent / "derived/matplotlib"
    destination = ensure_output_path(candidate)
    os.environ["MPLCONFIGDIR"] = str(destination)
    return destination
