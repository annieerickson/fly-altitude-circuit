"""Write guards for explicitly requested derived connectivity tables."""
from pathlib import Path
from ...paths import ensure_output_path


def output_directory(path):
    if path is None:
        raise ValueError('Regeneration requires an explicit writable output directory outside the data root.')
    destination = ensure_output_path(Path(path))
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def output_file(directory, filename):
    return ensure_output_path(output_directory(directory) / filename)
