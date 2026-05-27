"""
@file common.py
@author Mohamed EL BOUKHIARI
@brief Shared helpers for the GUI-adapted DCML implementation.

This module deliberately avoids any dependency on the original DCML repository
layout. Every path must be supplied by the caller or by the GUI.
"""

from __future__ import annotations

import logging
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Union

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - handled at runtime
    torch = None  # type: ignore[assignment]

PathLike = Union[str, os.PathLike[str]]
LOGGER_NAME = "DCML"


class BundleError(RuntimeError):
    """Raised when a model bundle cannot be serialized or loaded."""


@dataclass(frozen=True)
class DeviceInfo:
    """Runtime device information exposed to the GUI and summaries."""

    requested: str
    selected: str
    cuda_available: bool
    reason: str
    runtime_backend: str = "cpu"

    @property
    def torch_device(self) -> str:
        """Return a torch-compatible device string.

        DCML uses scikit-learn in this integration, so the actual estimator runs
        on CPU even when CUDA is visible.
        """
        return self.selected


def get_logger(name: str = LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    """Create or reuse a console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def ensure_dir(path: PathLike) -> Path:
    """Create a directory if needed and return its resolved path."""
    output = Path(path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    return output


def resolve_path(path: PathLike, base_dir: Optional[PathLike] = None) -> Path:
    """Resolve a file or directory path.

    Relative paths are resolved against ``base_dir`` when supplied, otherwise
    against the current working directory.
    """
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return path_obj.resolve()
    base = Path(base_dir).expanduser().resolve() if base_dir else Path.cwd().resolve()
    return (base / path_obj).resolve()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch when Torch is available."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def choose_device(requested: str | None = "cpu") -> DeviceInfo:
    """Normalize a device request.

    The returned value is honest about CUDA availability, but DCML remains a
    CPU backend because GradientBoostingRegressor is a scikit-learn estimator.
    """
    normalized = (requested or "cpu").strip().lower()
    allowed = {"auto", "cuda", "cpu", "cuda:0", "cuda:1"}
    if normalized not in allowed:
        raise ValueError("device must be one of: auto, cuda, cpu, cuda:0, cuda:1")

    cuda_available = bool(torch is not None and torch.cuda.is_available())
    if normalized == "cpu":
        return DeviceInfo(normalized, "cpu", cuda_available, "User requested CPU.")
    if normalized.startswith("cuda"):
        if cuda_available:
            return DeviceInfo(normalized, normalized, True, "CUDA requested and available; DCML still runs on CPU.")
        return DeviceInfo(normalized, "cpu", False, "CUDA requested but not available; DCML runs on CPU.")
    if cuda_available:
        return DeviceInfo(normalized, "cuda", True, "CUDA is available; DCML still runs on CPU.")
    return DeviceInfo(normalized, "cpu", False, "Auto-selected CPU because CUDA is not available.")


def cpu_only_warning(device_info: DeviceInfo, phase: str) -> str:
    """Return the standard CPU-only warning for DCML."""
    if device_info.requested.startswith("cuda"):
        return f"CUDA was requested, but DCML {phase} uses a scikit-learn backend and runs on CPU."
    if device_info.cuda_available:
        return f"CUDA is available, but DCML {phase} uses a scikit-learn backend and runs on CPU."
    return f"DCML {phase} uses a scikit-learn backend and runs on CPU."


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def serialize_estimator(estimator: Any) -> bytes:
    """Serialize a scikit-learn estimator to bytes."""
    try:
        return pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:  # pragma: no cover
        raise BundleError(f"Failed to serialize estimator: {exc}") from exc


def deserialize_estimator(blob: bytes) -> Any:
    """Deserialize a scikit-learn estimator from bytes."""
    try:
        return pickle.loads(blob)
    except Exception as exc:  # pragma: no cover
        raise BundleError(f"Failed to deserialize estimator: {exc}") from exc


def _require_torch() -> Any:
    if torch is None:
        raise BundleError("PyTorch is required to save/load .pt bundles, but it could not be imported.")
    return torch


def save_bundle(bundle: Mapping[str, Any], output_path: PathLike) -> Path:
    """Save a dictionary bundle to a ``.pt`` file using ``torch.save``."""
    torch_mod = _require_torch()
    output = resolve_path(output_path)
    ensure_dir(output.parent)
    try:
        torch_mod.save(dict(bundle), output)
    except Exception as exc:  # pragma: no cover
        raise BundleError(f"Failed to save bundle to {output}: {exc}") from exc
    return output


def load_bundle(bundle_path: PathLike, map_location: str = "cpu") -> MutableMapping[str, Any]:
    """Load a dictionary bundle created by :func:`save_bundle`."""
    torch_mod = _require_torch()
    path = resolve_path(bundle_path)
    if not path.is_file():
        raise BundleError(f"Bundle file does not exist: {path}")
    try:
        try:
            payload = torch_mod.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch_mod.load(path, map_location=map_location)
    except Exception as exc:  # pragma: no cover
        raise BundleError(f"Failed to load bundle from {path}: {exc}") from exc
    if not isinstance(payload, MutableMapping):
        raise BundleError("Loaded .pt file is not a dictionary bundle.")
    return payload


def build_training_bundle(
    *,
    estimator: Any,
    model_name: str,
    repo_variant: str,
    train_shape: tuple[int, int],
    feature_dtype: str,
    target_dtype: str,
    hyperparameters: Mapping[str, Any],
    seed: int,
    backend: str = "sklearn",
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Create the standardized DCML training bundle."""
    bundle: dict[str, Any] = {
        "model_name": model_name,
        "repo_variant": repo_variant,
        "backend": backend,
        "estimator_type": type(estimator).__name__,
        "serialized_estimator": serialize_estimator(estimator),
        "train_shape": list(train_shape),
        "n_features": int(train_shape[1]),
        "feature_dtype": feature_dtype,
        "target_dtype": target_dtype,
        "hyperparameters": dict(hyperparameters),
        "seed": int(seed),
        "training_timestamp": utc_now_iso(),
        "source_format": {
            "feature_container": "zip_with_single_npy",
            "label_container": "npy_1d",
        },
    }
    if extra_metadata:
        bundle.update(dict(extra_metadata))
    return bundle
