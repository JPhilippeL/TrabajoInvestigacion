"""
@file metrics_utils.py
@author Mohamed EL BOUKHIARI
@brief Metrics and plotting utilities for the GUI-adapted DCML module.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Optional, Sequence
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from DCML.Core.common import PathLike, resolve_path


class MetricsError(ValueError):
    """Raised when targets and predictions are incompatible."""


def _as_1d_float_array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise MetricsError(f"{name} must be a 1D array, got shape {array.shape}.")
    if array.size == 0:
        raise MetricsError(f"{name} must not be empty.")
    if np.isnan(array).any() or np.isinf(array).any():
        raise MetricsError(f"{name} contains NaN or infinite values.")
    return array


def _validate_true_pred(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    truth = _as_1d_float_array(y_true, "y_true")
    pred = _as_1d_float_array(y_pred, "y_pred")
    if truth.shape[0] != pred.shape[0]:
        raise MetricsError(
            f"y_true and y_pred must have the same length, got {truth.shape[0]} and {pred.shape[0]}."
        )
    return truth, pred


def rmse(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Compute root mean squared error."""
    truth, pred = _validate_true_pred(y_true, y_pred)
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def mae(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Compute mean absolute error."""
    truth, pred = _validate_true_pred(y_true, y_pred)
    return float(np.mean(np.abs(truth - pred)))


def pearson(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Compute Pearson correlation. Returns 0.0 when one input is constant."""
    truth, pred = _validate_true_pred(y_true, y_pred)
    if float(np.std(truth)) == 0.0 or float(np.std(pred)) == 0.0:
        return 0.0
    return float(np.corrcoef(truth, pred)[0, 1])


def compute_metrics(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Compute the standard DCML scalar metrics."""
    truth, pred = _validate_true_pred(y_true, y_pred)
    return {
        "RMSE": rmse(truth, pred),
        "Pearson": pearson(truth, pred),
        "MAE": mae(truth, pred),
    }


def build_metrics_rows(metrics: Mapping[str, float], std_default: float = 0.0) -> list[dict[str, float | str]]:
    """Convert a metrics dictionary into CSV-ready rows."""
    return [{"Metric": name, "Mean": float(value), "Std": float(std_default)} for name, value in metrics.items()]


def save_metrics_csv(metrics: Mapping[str, float], output_csv: PathLike, *, std_default: float = 0.0) -> Path:
    """Write metric rows to a CSV file."""
    path = resolve_path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_metrics_rows(metrics, std_default=std_default)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Metric", "Mean", "Std"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_predictions_csv(
    sample_ids: Sequence[str],
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    output_csv: PathLike,
) -> Path:
    """Write per-sample predictions to a CSV file."""
    truth, pred = _validate_true_pred(y_true, y_pred)
    if len(sample_ids) != truth.shape[0]:
        raise MetricsError(f"sample_ids length must match targets length, got {len(sample_ids)} and {truth.shape[0]}.")

    path = resolve_path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "true_affinity", "predicted_affinity"])
        writer.writeheader()
        for sample_id, true_value, pred_value in zip(sample_ids, truth, pred):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "true_affinity": float(true_value),
                    "predicted_affinity": float(pred_value),
                }
            )
    return path


def save_scatter_plot(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    output_png: PathLike,
    *,
    model_name: str = "DCML",
    split_id: Optional[str] = None,
    x_label: str = "Real value (pIC50)",
    y_label: str = "Predicted value (pIC50)",
    dpi: int = 200,
) -> Path:
    """Generate the standard prediction scatter plot."""
    truth, pred = _validate_true_pred(y_true, y_pred)
    metrics = compute_metrics(truth, pred)
    path = resolve_path(output_png)
    path.parent.mkdir(parents=True, exist_ok=True)

    lower = float(min(truth.min(), pred.min()))
    upper = float(max(truth.max(), pred.max()))
    if lower == upper:
        lower -= 1.0
        upper += 1.0

    split_suffix = f" - Split {split_id}" if split_id else ""
    title = f"{model_name}{split_suffix}\nRMSE = {metrics['RMSE']:.3f} | Pearson = {metrics['Pearson']:.3f}"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(truth, pred, alpha=0.7)
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
