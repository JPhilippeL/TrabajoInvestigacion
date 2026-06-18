"""Shared utilities for the TFM CAPLA implementation.

This module intentionally contains only lightweight helpers that can be reused by
training, debugging, and prediction scripts without pulling in model-specific or
heavy data-processing dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

PathLike = Union[str, os.PathLike, Path]


class CAPLAImplementationError(RuntimeError):
    """Base exception for implementation-specific runtime errors."""


class CAPLAPathError(CAPLAImplementationError):
    """Raised when a required path cannot be resolved."""


_LOGGER_CACHE = {}  # type: Dict[str, logging.Logger]


def get_logger(name: str = "TFM_CAPLA", level: int = logging.INFO) -> logging.Logger:
    """Return a configured stdout logger.

    Parameters
    ----------
    name:
        Logger name.
    level:
        Logging level.
    """
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    _LOGGER_CACHE[name] = logger
    return logger


def ensure_dir(path: PathLike) -> Path:
    """Create a directory if needed and return it as a resolved Path."""
    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_path(path: PathLike, base_dir: Optional[PathLike] = None, must_exist: bool = True) -> Path:
    """Resolve a path against an optional base directory.

    Relative paths are resolved against ``base_dir`` when provided, otherwise the
    current working directory is used.
    """
    p = Path(path).expanduser()

    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir).expanduser() / p

    p = p.resolve()

    if must_exist and not p.exists():
        raise CAPLAPathError(f"Path does not exist: {p}")

    return p


def find_capla_repo_root(start_path: Optional[PathLike] = None) -> Path:
    """Locate the project root containing CAPLA/original/src/capla.py.

    The function checks both:
    - the provided start path, when given;
    - otherwise, the current working directory and this module location.

    This makes the detection robust when scripts are launched from the GUI,
    from a terminal, or through ``python -m``.
    """
    marker = Path("CAPLA") / "original" / "src" / "capla.py"

    if start_path is None:
        starts = [
            Path.cwd(),
            Path(__file__).resolve(),
        ]
    else:
        starts = [
            Path(start_path).expanduser().resolve(),
            Path.cwd(),
            Path(__file__).resolve(),
        ]

    checked_paths: list[Path] = []

    for start in starts:
        current = start if start.is_dir() else start.parent

        for parent in [current, *current.parents]:
            candidate = parent / marker
            checked_paths.append(candidate)

            if candidate.exists():
                return parent

    checked_text = "\n".join(str(path) for path in checked_paths)

    raise CAPLAPathError(
        "Could not locate the CAPLA repository root. Expected to find "
        "'CAPLA/original/src/capla.py' from the current working directory, "
        "the provided start path, or the CAPLA module path.\n"
        f"Checked paths:\n{checked_text}"
    )


def choose_device(device: str = "auto") -> torch.device:
    """Select a torch device.

    Parameters
    ----------
    device:
        ``auto``, ``cuda``, or ``cpu``.
    """
    normalized = device.strip().lower()

    if normalized not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, cpu")

    if normalized == "cpu":
        return torch.device("cpu")

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise CAPLAImplementationError("CUDA was requested but is not available on this system.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(obj: Any, path: PathLike, indent: int = 2) -> Path:
    """Save a JSON-serializable object."""
    out = Path(path).expanduser().resolve()
    ensure_dir(out.parent)

    with out.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)

    return out


def load_json(path: PathLike) -> Any:
    """Load a JSON file."""
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_torch(obj: Any, path: PathLike) -> Path:
    """Save a PyTorch object with parent directory creation."""
    out = Path(path).expanduser().resolve()
    ensure_dir(out.parent)
    torch.save(obj, out)
    return out


def load_torch(path: PathLike, map_location: Optional[Union[str, torch.device]] = None) -> Any:
    """Load a PyTorch object."""
    return torch.load(Path(path).expanduser().resolve(), map_location=map_location)
