"""
@file data_utils.py
@author Mohamed EL BOUKHIARI
@brief Dataset loading and validation utilities for DCML.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np

from DCML.Core.common import PathLike, resolve_path

SampleIdMode = Literal["sample", "row_index"]


class DatasetValidationError(ValueError):
    """Raised when a DCML feature/label pair is malformed."""


@dataclass(frozen=True)
class DatasetValidationReport:
    """Summary statistics produced during validation."""

    n_samples: int
    n_features: int
    feature_dtype_original: str
    feature_dtype_final: str
    label_dtype_original: str
    label_dtype_final: str
    feature_nan_count: int
    feature_inf_count: int
    label_nan_count: int
    label_inf_count: int
    empty_row_count: int
    internal_npy_name: str


@dataclass(frozen=True)
class LoadedDataset:
    """In-memory representation of a validated DCML dataset."""

    features: np.ndarray
    labels: np.ndarray
    sample_ids: tuple[str, ...]
    feature_zip_path: Path
    label_npy_path: Path
    internal_npy_name: str
    report: DatasetValidationReport

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])


def create_sample_ids(n_samples: int, mode: SampleIdMode = "sample") -> tuple[str, ...]:
    """Create deterministic sample identifiers."""
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if mode == "sample":
        width = max(6, len(str(max(1, n_samples))))
        return tuple(f"sample_{idx:0{width}d}" for idx in range(1, n_samples + 1))
    if mode == "row_index":
        return tuple(str(idx) for idx in range(n_samples))
    raise ValueError("mode must be 'sample' or 'row_index'")


def _find_single_npy_in_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, mode="r") as archive:
        npy_members = [
            member
            for member in archive.namelist()
            if not member.endswith("/") and member.lower().endswith(".npy")
        ]
    if len(npy_members) != 1:
        raise DatasetValidationError(
            f"Expected exactly one .npy inside {zip_path}, found {len(npy_members)}."
        )
    return npy_members[0]


def _load_features_from_zip(zip_path: Path, internal_npy_name: str) -> np.ndarray:
    try:
        with zipfile.ZipFile(zip_path, mode="r") as archive:
            with archive.open(internal_npy_name, mode="r") as handle:
                data = handle.read()
        with io.BytesIO(data) as buffer:
            features = np.load(buffer, allow_pickle=False)
    except zipfile.BadZipFile as exc:
        raise DatasetValidationError(f"Feature ZIP is corrupt or unreadable: {zip_path}") from exc
    except ValueError as exc:
        raise DatasetValidationError(
            f"The internal NPY file could not be loaded safely from {zip_path}: {exc}"
        ) from exc
    except Exception as exc:
        raise DatasetValidationError(f"Failed to load features from {zip_path}: {exc}") from exc
    return features


def _load_labels(label_npy_path: Path) -> np.ndarray:
    try:
        labels = np.load(label_npy_path, allow_pickle=False)
    except ValueError as exc:
        raise DatasetValidationError(f"The label NPY could not be loaded safely: {exc}") from exc
    except Exception as exc:
        raise DatasetValidationError(f"Failed to load labels from {label_npy_path}: {exc}") from exc
    return labels


def _validate_feature_label_arrays(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    internal_npy_name: str,
    feature_dtype_original: str,
    label_dtype_original: str,
) -> DatasetValidationReport:
    if features.ndim != 2:
        raise DatasetValidationError(f"Feature matrix must be 2D, got shape {features.shape}.")
    if labels.ndim != 1:
        raise DatasetValidationError(f"Label array must be 1D, got shape {labels.shape}.")
    if features.shape[0] != labels.shape[0]:
        raise DatasetValidationError(
            "Number of feature rows does not match number of labels: "
            f"{features.shape[0]} != {labels.shape[0]}."
        )
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise DatasetValidationError("Feature matrix must contain at least one sample and one feature.")
    if not np.issubdtype(features.dtype, np.number):
        raise DatasetValidationError(f"Feature matrix must be numeric, got dtype {features.dtype}.")
    if not np.issubdtype(labels.dtype, np.number):
        raise DatasetValidationError(f"Label array must be numeric, got dtype {labels.dtype}.")

    feature_nan_count = int(np.isnan(features).sum())
    feature_inf_count = int(np.isinf(features).sum())
    label_nan_count = int(np.isnan(labels).sum())
    label_inf_count = int(np.isinf(labels).sum())
    empty_row_count = int(np.all(features == 0, axis=1).sum())

    if feature_nan_count:
        raise DatasetValidationError(f"Feature matrix contains {feature_nan_count} NaN values.")
    if feature_inf_count:
        raise DatasetValidationError(f"Feature matrix contains {feature_inf_count} infinite values.")
    if label_nan_count:
        raise DatasetValidationError(f"Label array contains {label_nan_count} NaN values.")
    if label_inf_count:
        raise DatasetValidationError(f"Label array contains {label_inf_count} infinite values.")
    if empty_row_count:
        raise DatasetValidationError(f"Feature matrix contains {empty_row_count} completely empty rows.")

    return DatasetValidationReport(
        n_samples=int(features.shape[0]),
        n_features=int(features.shape[1]),
        feature_dtype_original=feature_dtype_original,
        feature_dtype_final=str(features.dtype),
        label_dtype_original=label_dtype_original,
        label_dtype_final=str(labels.dtype),
        feature_nan_count=feature_nan_count,
        feature_inf_count=feature_inf_count,
        label_nan_count=label_nan_count,
        label_inf_count=label_inf_count,
        empty_row_count=empty_row_count,
        internal_npy_name=internal_npy_name,
    )


def load_dcml_dataset(
    *,
    feature_zip: PathLike,
    label_npy: PathLike,
    cast_float32: bool = False,
    label_dtype: Optional[np.dtype] = None,
    sample_id_mode: SampleIdMode = "sample",
    base_dir: Optional[PathLike] = None,
) -> LoadedDataset:
    """Load and validate a DCML dataset from ``feature.zip`` and ``label.npy``."""
    feature_zip_path = resolve_path(feature_zip, base_dir=base_dir)
    label_npy_path = resolve_path(label_npy, base_dir=base_dir)

    if not feature_zip_path.is_file():
        raise DatasetValidationError(f"Feature ZIP does not exist: {feature_zip_path}")
    if not label_npy_path.is_file():
        raise DatasetValidationError(f"Label NPY does not exist: {label_npy_path}")

    internal_npy_name = _find_single_npy_in_zip(feature_zip_path)
    features = _load_features_from_zip(feature_zip_path, internal_npy_name)
    labels = _load_labels(label_npy_path)

    feature_dtype_original = str(features.dtype)
    label_dtype_original = str(labels.dtype)

    if cast_float32:
        features = features.astype(np.float32, copy=False)
    if label_dtype is not None:
        labels = labels.astype(label_dtype, copy=False)

    report = _validate_feature_label_arrays(
        features=features,
        labels=labels,
        internal_npy_name=internal_npy_name,
        feature_dtype_original=feature_dtype_original,
        label_dtype_original=label_dtype_original,
    )

    return LoadedDataset(
        features=features,
        labels=labels,
        sample_ids=create_sample_ids(report.n_samples, mode=sample_id_mode),
        feature_zip_path=feature_zip_path,
        label_npy_path=label_npy_path,
        internal_npy_name=internal_npy_name,
        report=report,
    )


def create_debug_subset(dataset: LoadedDataset, *, max_samples: int = 2000, seed: int = 42) -> LoadedDataset:
    """Return a deterministic subset for fast debug runs."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if dataset.n_samples <= max_samples:
        return dataset

    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(dataset.n_samples, size=max_samples, replace=False))
    subset_features = dataset.features[selected]
    subset_labels = dataset.labels[selected]
    subset_ids = tuple(dataset.sample_ids[int(idx)] for idx in selected)

    report = _validate_feature_label_arrays(
        features=subset_features,
        labels=subset_labels,
        internal_npy_name=dataset.internal_npy_name,
        feature_dtype_original=dataset.report.feature_dtype_original,
        label_dtype_original=dataset.report.label_dtype_original,
    )

    return LoadedDataset(
        features=subset_features,
        labels=subset_labels,
        sample_ids=subset_ids,
        feature_zip_path=dataset.feature_zip_path,
        label_npy_path=dataset.label_npy_path,
        internal_npy_name=dataset.internal_npy_name,
        report=report,
    )


def save_feature_zip(
    features: np.ndarray,
    output_zip: PathLike,
    *,
    internal_npy_name: str = "features.npy",
    compressed: bool = True,
) -> Path:
    """Save a 2D feature matrix into a DCML-compatible ZIP."""
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    output_path = resolve_path(output_zip)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with io.BytesIO() as buffer:
        np.save(buffer, features, allow_pickle=False)
        payload = buffer.getvalue()
    with zipfile.ZipFile(output_path, mode="w", compression=compression) as archive:
        archive.writestr(internal_npy_name, payload)
    return output_path


def save_labels_npy(labels: np.ndarray, output_npy: PathLike) -> Path:
    """Save a 1D label array as ``.npy``."""
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    output_path = resolve_path(output_npy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, labels, allow_pickle=False)
    return output_path
