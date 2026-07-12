"""
@file dcml_trainer.py
@author Mohamed EL BOUKHIARI
@brief Training pipeline for the GUI-adapted DCML module.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import GradientBoostingRegressor

from DCML.Core.common import (
    PathLike,
    build_training_bundle,
    choose_device,
    cpu_only_warning,
    ensure_dir,
    get_logger,
    resolve_path,
    save_bundle,
    seed_everything,
    utc_now_iso,
)
from DCML.Core.data_utils import DatasetValidationError, LoadedDataset, load_dcml_dataset
from DCML.Core.metrics_utils import compute_metrics

LOGGER = get_logger(__name__)

_NUMPY_MEMORY_ERROR = getattr(getattr(np, "_core", object()), "_exceptions", None)
if _NUMPY_MEMORY_ERROR is not None and hasattr(_NUMPY_MEMORY_ERROR, "_ArrayMemoryError"):
    NUMPY_MEMORY_EXCEPTIONS: tuple[type[BaseException], ...] = (MemoryError, _NUMPY_MEMORY_ERROR._ArrayMemoryError)
else:
    NUMPY_MEMORY_EXCEPTIONS = (MemoryError,)


class TrainDCMLError(RuntimeError):
    """Raised when DCML training fails."""


DEFAULT_MODEL_TYPE = "gradient_boosting"
DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.01,
    "min_samples_split": 2,
    "subsample": 0.7,
    "max_features": "sqrt",
    "loss": "squared_error",
}


def _normalize_loss_alias(loss: str) -> str:
    normalized = str(loss).strip().lower()
    aliases = {
        "ls": "squared_error",
        "squared_error": "squared_error",
        "lad": "absolute_error",
        "absolute_error": "absolute_error",
        "huber": "huber",
        "quantile": "quantile",
    }
    if normalized not in aliases:
        raise ValueError("Unsupported loss. Use: squared_error, absolute_error, huber, quantile.")
    return aliases[normalized]


def _parse_max_features(raw_value: Any) -> str | int | float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, (int, np.integer)):
        value = int(raw_value)
        if value <= 0:
            raise ValueError("max_features integer must be positive.")
        return value
    if isinstance(raw_value, (float, np.floating)):
        value = float(raw_value)
        if not (0.0 < value <= 1.0):
            raise ValueError("max_features float must be in (0, 1].")
        return value

    value = str(raw_value).strip().lower()
    if value in {"none", "null", ""}:
        return None
    if value in {"sqrt", "log2"}:
        return value
    try:
        if any(char in value for char in [".", "e"]):
            numeric = float(value)
            if not (0.0 < numeric <= 1.0):
                raise ValueError
            return numeric
        integer_value = int(value)
        if integer_value <= 0:
            raise ValueError
        return integer_value
    except ValueError as exc:
        raise ValueError("Invalid max_features. Use sqrt, log2, none, positive integer, or float in (0, 1].") from exc


def validate_hyperparameters(hyperparameters: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Merge and validate DCML/GradientBoosting hyperparameters."""
    merged = dict(DEFAULT_HYPERPARAMETERS)
    if hyperparameters:
        merged.update(dict(hyperparameters))

    n_estimators = int(merged["n_estimators"])
    max_depth = int(merged["max_depth"])
    learning_rate = float(merged["learning_rate"])
    min_samples_split = int(merged["min_samples_split"])
    subsample = float(merged["subsample"])

    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive.")
    if max_depth <= 0:
        raise ValueError("max_depth must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if min_samples_split < 2:
        raise ValueError("min_samples_split must be at least 2.")
    if not (0.0 < subsample <= 1.0):
        raise ValueError("subsample must be in (0, 1].")

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "min_samples_split": min_samples_split,
        "subsample": subsample,
        "max_features": _parse_max_features(merged["max_features"]),
        "loss": _normalize_loss_alias(str(merged["loss"])),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _estimate_array_size_mb(array: np.ndarray) -> float:
    return float(array.nbytes / (1024.0 * 1024.0))


def build_estimator(model_type: str, hyperparameters: Mapping[str, Any], seed: int) -> GradientBoostingRegressor:
    """Build the DCML estimator."""
    if model_type != DEFAULT_MODEL_TYPE:
        raise TrainDCMLError(f"Unsupported model type for DCML: {model_type}")
    return GradientBoostingRegressor(random_state=int(seed), **dict(hyperparameters))


def train_model(dataset: LoadedDataset, estimator: GradientBoostingRegressor) -> tuple[GradientBoostingRegressor, dict[str, float], float]:
    """Fit the estimator and return train metrics with elapsed seconds."""
    start = time.perf_counter()
    estimator.fit(dataset.features, dataset.labels)
    elapsed = time.perf_counter() - start
    predictions = estimator.predict(dataset.features)
    train_metrics = compute_metrics(dataset.labels, predictions)
    return estimator, train_metrics, float(elapsed)


def train_dcml(
    *,
    train_feature_zip: PathLike,
    train_label_npy: PathLike,
    output_model: PathLike,
    output_dir: PathLike,
    model_type: str = DEFAULT_MODEL_TYPE,
    hyperparameters: Optional[Mapping[str, Any]] = None,
    device: str | None = "cpu",
    seed: int = 42,
    cast_float32: bool = True,
    progress_callback=None,
) -> dict[str, Any]:
    """Train DCML from a feature ZIP and label NPY.

    Returns the same summary dictionary that is written to
    ``training_summary.json``.
    """
    seed_everything(int(seed))
    feature_zip_path = resolve_path(train_feature_zip)
    label_npy_path = resolve_path(train_label_npy)
    output_model_path = resolve_path(output_model)
    output_dir_path = ensure_dir(output_dir)
    ensure_dir(output_model_path.parent)
    if progress_callback:
        progress_callback("Loading training features and labels.")
        progress_callback(f"Feature ZIP: {feature_zip_path}")
        progress_callback(f"Label NPY: {label_npy_path}")

    device_info = choose_device(device)
    warning = cpu_only_warning(device_info, "training")
    LOGGER.warning(warning)
    if progress_callback:
        progress_callback("Warning: " + warning)

    hparams = validate_hyperparameters(hyperparameters)
    if progress_callback:
        progress_callback(f"Validated hyperparameters: {hparams}")

    dataset = load_dcml_dataset(
        feature_zip=feature_zip_path,
        label_npy=label_npy_path,
        cast_float32=bool(cast_float32),
    )
    if progress_callback:
        progress_callback(
            f"Loaded training feature shape: {tuple(dataset.features.shape)}, dtype={dataset.report.feature_dtype_final}"
        )
        progress_callback(
            f"Loaded training label shape: {tuple(dataset.labels.shape)}, dtype={dataset.report.label_dtype_final}"
        )

    config_payload = {
        "script": "dcml_trainer.py",
        "timestamp_utc": utc_now_iso(),
        "inputs": {
            "train_feature_zip": str(feature_zip_path),
            "train_label_npy": str(label_npy_path),
        },
        "outputs": {
            "output_model": str(output_model_path),
            "output_dir": str(output_dir_path),
        },
        "device": {
            "requested": device_info.requested,
            "interface_selected": device_info.selected,
            "runtime_backend": "cpu",
        },
        "model": {
            "model_type": model_type,
            "backend": "sklearn",
            "estimator_type": "GradientBoostingRegressor",
            "hyperparameters": hparams,
        },
        "data": {"cast_float32": bool(cast_float32)},
        "seed": int(seed),
    }
    _write_json(output_dir_path / "training_config.json", config_payload)

    estimator = build_estimator(model_type, hparams, int(seed))
    try:
        if progress_callback:
            progress_callback("Training started.")
        estimator, train_metrics, training_seconds = train_model(dataset, estimator)
        if progress_callback:
            progress_callback(f"Training finished in {training_seconds:.3f}s.")
            progress_callback(f"Train metrics: {train_metrics}")
    except NUMPY_MEMORY_EXCEPTIONS as exc:
        raise TrainDCMLError(
            "Training ran out of memory. Enable cast_float32 or reduce n_estimators/max_depth."
        ) from exc

    extra_metadata = {
        "sklearn_version": sklearn_version,
        "notes": "GUI-adapted DCML artifact. Backend is scikit-learn and runtime is CPU-only.",
        "training_metrics": {name: float(value) for name, value in train_metrics.items()},
        "device_request": device_info.requested,
        "runtime_backend_device": "cpu",
    }
    bundle = build_training_bundle(
        estimator=estimator,
        model_name="DCML",
        repo_variant="DCML_GUI",
        train_shape=(dataset.n_samples, dataset.n_features),
        feature_dtype=dataset.report.feature_dtype_final,
        target_dtype=dataset.report.label_dtype_final,
        hyperparameters=hparams,
        seed=int(seed),
        backend="sklearn",
        extra_metadata=extra_metadata,
    )
    save_bundle(bundle, output_model_path)
    if progress_callback:
        progress_callback(f"Model checkpoint written: {output_model_path}")

    bundle_size_mb = output_model_path.stat().st_size / (1024.0 * 1024.0) if output_model_path.exists() else None
    summary_payload: dict[str, Any] = {
        "script": "dcml_trainer.py",
        "timestamp_utc": utc_now_iso(),
        "status": "success",
        "backend": "sklearn",
        "estimator_type": "GradientBoostingRegressor",
        "device_warning": warning,
        "inputs": {
            "train_feature_zip": str(feature_zip_path),
            "train_label_npy": str(label_npy_path),
        },
        "dataset": {
            "n_samples": int(dataset.n_samples),
            "n_features": int(dataset.n_features),
            "feature_dtype_original": dataset.report.feature_dtype_original,
            "feature_dtype_final": dataset.report.feature_dtype_final,
            "label_dtype_original": dataset.report.label_dtype_original,
            "label_dtype_final": dataset.report.label_dtype_final,
            "internal_npy_name": dataset.internal_npy_name,
            "feature_size_mb": round(_estimate_array_size_mb(dataset.features), 3),
            "label_size_mb": round(_estimate_array_size_mb(dataset.labels), 6),
        },
        "hyperparameters": hparams,
        "training": {
            "training_seconds": float(training_seconds),
            "train_metrics": {name: float(value) for name, value in train_metrics.items()},
        },
        "artifact": {
            "output_model": str(output_model_path),
            "artifact_size_mb": round(bundle_size_mb, 3) if bundle_size_mb is not None else None,
        },
        "outputs": {
            "training_config_json": str(output_dir_path / "training_config.json"),
            "training_summary_json": str(output_dir_path / "training_summary.json"),
        },
        "versions": {"sklearn": sklearn_version, "numpy": np.__version__},
    }
    _write_json(output_dir_path / "training_summary.json", summary_payload)
    if progress_callback:
        progress_callback(f"Training summary written: {output_dir_path / 'training_summary.json'}")
    return summary_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DCML from feature.zip + label.npy.")
    parser.add_argument("--train-feature-zip", required=True)
    parser.add_argument("--train-label-npy", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cast-float32", action="store_true")
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE, choices=[DEFAULT_MODEL_TYPE])
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_HYPERPARAMETERS["n_estimators"])
    parser.add_argument("--max-depth", type=int, default=DEFAULT_HYPERPARAMETERS["max_depth"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_HYPERPARAMETERS["learning_rate"])
    parser.add_argument("--min-samples-split", type=int, default=DEFAULT_HYPERPARAMETERS["min_samples_split"])
    parser.add_argument("--subsample", type=float, default=DEFAULT_HYPERPARAMETERS["subsample"])
    parser.add_argument("--max-features", default=str(DEFAULT_HYPERPARAMETERS["max_features"]))
    parser.add_argument("--loss", default=str(DEFAULT_HYPERPARAMETERS["loss"]))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        hparams = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "min_samples_split": args.min_samples_split,
            "subsample": args.subsample,
            "max_features": args.max_features,
            "loss": args.loss,
        }
        summary = train_dcml(
            train_feature_zip=args.train_feature_zip,
            train_label_npy=args.train_label_npy,
            output_model=args.output_model,
            output_dir=args.output_dir,
            model_type=args.model_type,
            hyperparameters=hparams,
            device=args.device,
            seed=args.seed,
            cast_float32=args.cast_float32,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except (DatasetValidationError, TrainDCMLError, ValueError, FileNotFoundError) as exc:
        LOGGER.error(str(exc))
        return 1
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Unexpected DCML training failure: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
