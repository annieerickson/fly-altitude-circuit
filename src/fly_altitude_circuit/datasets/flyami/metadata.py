"""Driver labels and display IDs from the screen-line documentation."""

from ...config import load


def line_label(line, abbrev=False):
    e = load("screen_lines")["lines"][line]
    return e.get("driver_abbrev", e["driver"]) if abbrev else e["driver"]

def line_names(line, abbrev=False):
    """Return (cell_type, line_id), e.g. ('DNg03', '2548')."""
    e = load("screen_lines")["lines"][line]
    cell_type = e.get("driver_abbrev", e["driver"]) if abbrev else e["driver"]
    return cell_type, e["id"]
