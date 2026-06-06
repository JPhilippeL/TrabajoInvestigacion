"""Metrics and output helpers for CAPLA experiments."""

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ensure_dir


class MetricsError(ValueError):
    """Raised when metrics cannot be computed."""


def _as_1d_float_array(values: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise MetricsError("Cannot compute metrics on an empty array.")
    return arr


def rmse(y_true: Union[Sequence[float], np.ndarray], y_pred: Union[Sequence[float], np.ndarray]) -> float:
    y_true_arr = _as_1d_float_array(y_true)
    y_pred_arr = _as_1d_float_array(y_pred)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def mae(y_true: Union[Sequence[float], np.ndarray], y_pred: Union[Sequence[float], np.ndarray]) -> float:
    y_true_arr = _as_1d_float_array(y_true)
    y_pred_arr = _as_1d_float_array(y_pred)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def pearson(y_true: Union[Sequence[float], np.ndarray], y_pred: Union[Sequence[float], np.ndarray]) -> float:
    y_true_arr = _as_1d_float_array(y_true)
    y_pred_arr = _as_1d_float_array(y_pred)
    if y_true_arr.size < 2:
        return float("nan")
    true_std = float(np.std(y_true_arr))
    pred_std = float(np.std(y_pred_arr))
    if true_std == 0.0 or pred_std == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true_arr, y_pred_arr)[0, 1])


def sd(y_true: Union[Sequence[float], np.ndarray], y_pred: Union[Sequence[float], np.ndarray]) -> float:
    """Compute the CAPLA-style standard deviation metric.

    This matches the original implementation conceptually, but avoids the sklearn
    dependency by using a least-squares linear fit.
    """
    y_true_arr = _as_1d_float_array(y_true)
    y_pred_arr = _as_1d_float_array(y_pred)
    if y_true_arr.size < 2:
        return float("nan")
    X = np.column_stack([y_pred_arr, np.ones_like(y_pred_arr)])
    coef, _, _, _ = np.linalg.lstsq(X, y_true_arr, rcond=None)
    fitted = X @ coef
    return float(np.sqrt(np.square(y_true_arr - fitted).sum() / max(len(y_true_arr) - 1, 1)))


def compute_regression_metrics(
    y_true: Union[Sequence[float], np.ndarray],
    y_pred: Union[Sequence[float], np.ndarray],
) -> Dict[str, float]:
    """Compute the regression metrics used throughout the implementation."""
    return {
        "RMSE": rmse(y_true, y_pred),
        "Pearson": pearson(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "SD": sd(y_true, y_pred),
    }


def save_metrics_csv(
    metrics: Mapping[str, float],
    path: Union[str, Path],
    std_overrides: Optional[Mapping[str, float]] = None,
) -> Path:
    """Save a metric dictionary using the expected academic format."""
    std_overrides = dict(std_overrides or {})
    rows = [
        {"Metric": name, "Mean": float(value), "Std": float(std_overrides.get(name, 0.0))}
        for name, value in metrics.items()
    ]
    out = Path(path).expanduser().resolve()
    ensure_dir(out.parent)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def save_predictions_csv(
    pdbids: Sequence[str],
    y_true: Union[Sequence[float], np.ndarray],
    y_pred: Union[Sequence[float], np.ndarray],
    path: Union[str, Path],
) -> Path:
    """Save per-complex predictions."""
    out = Path(path).expanduser().resolve()
    ensure_dir(out.parent)
    df = pd.DataFrame(
        {
            "pdbid": list(pdbids),
            "true_affinity": _as_1d_float_array(y_true),
            "predicted_affinity": _as_1d_float_array(y_pred),
        }
    )
    df.to_csv(out, index=False)
    return out


def save_scatter_plot(
    y_true: Union[Sequence[float], np.ndarray],
    y_pred: Union[Sequence[float], np.ndarray],
    path: Union[str, Path],
    model_name: str,
    split_id: Optional[str] = None,
    metrics: Optional[Mapping[str, float]] = None,
) -> Path:
    """Save the real-vs-predicted scatter plot."""
    y_true_arr = _as_1d_float_array(y_true)
    y_pred_arr = _as_1d_float_array(y_pred)
    metrics = dict(metrics or compute_regression_metrics(y_true_arr, y_pred_arr))

    out = Path(path).expanduser().resolve()
    ensure_dir(out.parent)

    min_val = float(min(y_true_arr.min(), y_pred_arr.min()))
    max_val = float(max(y_true_arr.max(), y_pred_arr.max()))
    padding = max((max_val - min_val) * 0.05, 1e-6)

    title = model_name if split_id is None else f"{model_name} - Split {split_id}"
    subtitle = f"RMSE = {metrics['RMSE']:.3f} | Pearson = {metrics['Pearson']:.3f}"

    plt.figure(figsize=(8, 7))
    plt.scatter(y_true_arr, y_pred_arr, alpha=0.7)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlim(min_val - padding, max_val + padding)
    plt.ylim(min_val - padding, max_val + padding)
    plt.xlabel("Valor real (pIC50)")
    plt.ylabel("Valor predicho (pIC50)")
    plt.title(f"{title}\n{subtitle}")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out
