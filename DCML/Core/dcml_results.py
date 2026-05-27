"""
@file dcml_results.py
@author Mohamed EL BOUKHIARI
@brief Result parsing helpers for the DCML module.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from DCML.Core.common import PathLike, resolve_path


def read_json(path: PathLike) -> dict[str, Any]:
    """Read a JSON file as a dictionary."""
    resolved = resolve_path(path)
    return json.loads(resolved.read_text(encoding="utf-8"))


def read_metrics_csv(path: PathLike) -> dict[str, float]:
    """Read a DCML metrics CSV produced by ``save_metrics_csv``."""
    resolved = resolve_path(path)
    metrics: dict[str, float] = {}
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric = row.get("Metric")
            value = row.get("Mean")
            if metric and value is not None:
                metrics[metric] = float(value)
    return metrics


def format_seconds(seconds: float) -> str:
    """Format elapsed seconds for GUI messages."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"


def ensure_parent(path: PathLike) -> Path:
    """Create parent directory for a file path and return the resolved path."""
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
