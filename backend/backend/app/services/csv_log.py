"""Append-only CSV logs under logs/ (Gemini costs, try-on timings)."""
import csv
import os
from datetime import datetime
from typing import Any, Dict, List


def append_row(path: str, fields: List[str], row: Dict[str, Any]) -> None:
    """
    Appends one row, writing the header first when the file is new. A file written with an older column set is kept
    aside as <name>_until_<timestamp>.csv and a new file is started, so columns never shift under existing rows.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), None)
        if header is not None and header != fields:
            stem, ext = os.path.splitext(path)
            os.replace(path, f"{stem}_until_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}")
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
