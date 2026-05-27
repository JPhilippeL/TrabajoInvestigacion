"""
@file dcml_tester.py
@author Mohamed EL BOUKHIARI
@brief Evaluation pipeline for the GUI-adapted DCML module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import __version__ as sklearn_version

from DCML.Core.common import (
    BundleError,
    PathLike,
    choose_device,
    cpu_only_warning,
    deserialize_estimator,
    ensure_dir,
    get_logger,
    load_bundle,
    resolve_path,
    utc_now_iso,
)
from DCML.Core.data_utils import DatasetValidationError, LoadedDataset, load_dcml_dataset
from DCML.Core.metrics_utils import compute_metrics, save_metrics_csv, save_predictions_csv, save_scatter_plot

LOGGER = get_logger(__name__)


class PredictDCMLError(RuntimeError):
    """Raised when DCML prediction/evaluation fails."""


REQUIRED_BUNDLE_KEYS = {
    "model_name",
    "repo_variant",
    "backend",
    "estimator_type",
    "serialized_estimator",
    "n_features",
}


def _split_suffix(split_id: str | None) -> str:
    """
    Build a clean optional suffix for output files.

    Examples
    --------
    split_id="smoke" -> "_smoke"
    split_id="gui_smoke" -> "_gui_smoke"
    split_id="trial_0001" -> "_trial_0001"
    """
    if split_id is None:
        return ""

    clean_id = str(split_id).strip()
    if not clean_id:
        return ""

    clean_id = clean_id.replace(" ", "_")
    return f"_{clean_id}"


def _validate_bundle(bundle: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_BUNDLE_KEYS.difference(bundle.keys()))
    if missing:
        raise PredictDCMLError("The provided .pt bundle is incomplete. Missing keys: " + ", ".join(missing))
    if bundle.get("backend") != "sklearn":
        raise PredictDCMLError(f"Unsupported backend in bundle: {bundle.get('backend')!r}. Expected 'sklearn'.")
    if not isinstance(bundle.get("serialized_estimator"), (bytes, bytearray)):
        raise PredictDCMLError("The bundle does not contain a valid serialized estimator blob.")
    try:
        n_features = int(bundle.get("n_features"))
    except Exception as exc:
        raise PredictDCMLError("The bundle field 'n_features' is missing or invalid.") from exc
    if n_features <= 0:
        raise PredictDCMLError("The bundle field 'n_features' must be positive.")


def load_estimator_from_bundle(bundle_path: PathLike) -> tuple[dict[str, Any], Any]:
    """Load a DCML .pt bundle and reconstruct the serialized estimator."""
    try:
        raw_bundle = load_bundle(bundle_path, map_location="cpu")
    except BundleError as exc:
        raise PredictDCMLError(str(exc)) from exc
    bundle = dict(raw_bundle)
    _validate_bundle(bundle)
    try:
        estimator = deserialize_estimator(bundle["serialized_estimator"])
    except BundleError as exc:
        raise PredictDCMLError(str(exc)) from exc
    if not hasattr(estimator, "predict"):
        raise PredictDCMLError("The estimator reconstructed from the bundle has no predict() method.")
    return bundle, estimator


def _resolve_outputs(output_dir: Path, split_id: str | None) -> tuple[Path, Path, Path, Path]:
    suffix = _split_suffix(split_id)
    predictions_csv = output_dir / f"Predictions_DCML{suffix}.csv"
    metrics_csv = output_dir / f"Metrics_DCML{suffix}.csv"
    scatter_png = output_dir / f"Scatter_DCML{suffix}.png"
    summary_json = output_dir / f"prediction_summary{suffix}.json"
    return predictions_csv, metrics_csv, scatter_png, summary_json


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _check_feature_compatibility(bundle: dict[str, Any], estimator: Any, dataset: LoadedDataset) -> None:
    expected_n_features = int(bundle["n_features"])
    if dataset.n_features != expected_n_features:
        raise PredictDCMLError(
            "Feature dimensionality mismatch between model bundle and dataset: "
            f"model expects {expected_n_features}, dataset provides {dataset.n_features}."
        )
    estimator_n_features = getattr(estimator, "n_features_in_", None)
    if estimator_n_features is not None and int(estimator_n_features) != dataset.n_features:
        raise PredictDCMLError(
            "The deserialized estimator reports a different number of features than the dataset: "
            f"estimator has {int(estimator_n_features)}, dataset provides {dataset.n_features}."
        )


def run_prediction(estimator: Any, dataset: LoadedDataset) -> np.ndarray:
    """Run inference and return a 1D prediction array."""
    try:
        predictions = estimator.predict(dataset.features)
    except MemoryError as exc:
        raise PredictDCMLError("Inference ran out of memory. Enable cast_float32 or use more RAM.") from exc
    except ValueError as exc:
        raise PredictDCMLError(f"Estimator prediction failed: {exc}") from exc
    except Exception as exc:
        raise PredictDCMLError(f"Unexpected error during prediction: {exc}") from exc

    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim != 1:
        raise PredictDCMLError(f"Estimator returned invalid prediction shape {predictions.shape}; expected 1D.")
    if predictions.shape[0] != dataset.n_samples:
        raise PredictDCMLError(
            "Estimator returned a different number of predictions than input samples: "
            f"{predictions.shape[0]} != {dataset.n_samples}."
        )
    if np.isnan(predictions).any() or np.isinf(predictions).any():
        raise PredictDCMLError("Predictions contain NaN or infinite values.")
    return predictions


def test_dcml(
    *,
    model_pt: PathLike,
    feature_zip: PathLike,
    label_npy: PathLike,
    output_dir: PathLike,
    device: str | None = "cpu",
    split_id: str | None = None,
    dataset_name: str | None = None,
    cast_float32: bool = True,
) -> dict[str, Any]:
    """Evaluate a trained DCML bundle on an external dataset."""
    model_path = resolve_path(model_pt)
    feature_zip_path = resolve_path(feature_zip)
    label_npy_path = resolve_path(label_npy)
    output_dir_path = ensure_dir(output_dir)

    device_info = choose_device(device)
    warning = cpu_only_warning(device_info, "inference")
    LOGGER.warning(warning)

    bundle, estimator = load_estimator_from_bundle(model_path)
    dataset = load_dcml_dataset(
        feature_zip=feature_zip_path,
        label_npy=label_npy_path,
        cast_float32=bool(cast_float32),
        sample_id_mode="row_index",
    )
    _check_feature_compatibility(bundle, estimator, dataset)

    predictions = run_prediction(estimator, dataset)
    metrics = compute_metrics(dataset.labels, predictions)

    predictions_csv, metrics_csv, scatter_png, summary_json = _resolve_outputs(output_dir_path, split_id)
    save_predictions_csv(dataset.sample_ids, dataset.labels, predictions, predictions_csv)
    save_metrics_csv(metrics, metrics_csv, std_default=0.0)
    save_scatter_plot(dataset.labels, predictions, scatter_png, model_name="DCML", split_id=split_id)

    summary = {
        "script": "dcml_tester.py",
        "timestamp_utc": utc_now_iso(),
        "status": "success",
        "model": {
            "model_pt": str(model_path),
            "model_name": bundle.get("model_name"),
            "repo_variant": bundle.get("repo_variant"),
            "backend": bundle.get("backend"),
            "estimator_type": bundle.get("estimator_type"),
            "n_features_expected": int(bundle.get("n_features")),
            "feature_dtype_trained": bundle.get("feature_dtype"),
            "target_dtype_trained": bundle.get("target_dtype"),
            "training_timestamp": bundle.get("training_timestamp"),
            "training_metrics": bundle.get("training_metrics"),
            "hyperparameters": bundle.get("hyperparameters"),
        },
        "inputs": {
            "feature_zip": str(feature_zip_path),
            "label_npy": str(label_npy_path),
            "dataset_name": dataset_name,
            "split_id": split_id,
        },
        "dataset": {
            "n_samples": int(dataset.n_samples),
            "n_features": int(dataset.n_features),
            "feature_dtype_original": dataset.report.feature_dtype_original,
            "feature_dtype_final": dataset.report.feature_dtype_final,
            "label_dtype_original": dataset.report.label_dtype_original,
            "label_dtype_final": dataset.report.label_dtype_final,
            "internal_npy_name": dataset.internal_npy_name,
        },
        "metrics": {name: float(value) for name, value in metrics.items()},
        "device_warning": warning,
        "outputs": {
            "predictions_csv": str(predictions_csv),
            "metrics_csv": str(metrics_csv),
            "scatter_png": str(scatter_png),
            "summary_json": str(summary_json),
        },
        "versions": {"numpy": np.__version__, "sklearn": sklearn_version},
    }
    _write_json(summary_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained DCML .pt bundle.")
    parser.add_argument("--model-pt", required=True)
    parser.add_argument("--feature-zip", required=True)
    parser.add_argument("--label-npy", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split-id", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--cast-float32", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = test_dcml(
            model_pt=args.model_pt,
            feature_zip=args.feature_zip,
            label_npy=args.label_npy,
            output_dir=args.output_dir,
            device=args.device,
            split_id=args.split_id,
            dataset_name=args.dataset_name,
            cast_float32=args.cast_float32,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except (PredictDCMLError, DatasetValidationError, FileNotFoundError, ValueError) as exc:
        LOGGER.error(str(exc))
        return 1
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Unexpected DCML prediction failure: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
