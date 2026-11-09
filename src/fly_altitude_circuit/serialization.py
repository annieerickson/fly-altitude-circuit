"""Export analysis reports."""
import json
import numpy as np
from .paths import ensure_output_path

def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_analysis_report(path, payload, *, has_statistics=False):
    """Save statistics or printed analysis values
    """
    if not has_statistics and not any(str(line).strip() for line in payload.get("console", [])):
        return False
    destination = ensure_output_path(path)
    # Configuration lives in YAML; saved reports contain analysis output only.
    report = {key: value for key, value in payload.items() if key != "parameters"}
    destination.write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False) + "\n")
    return True

