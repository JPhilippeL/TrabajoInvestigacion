#!/usr/bin/env python3
"""Create diagnostic tables for MPro-URV DCML predictions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def rmse(x: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(x.to_numpy(dtype=float)))))


def pearson(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a.to_numpy(dtype=float), b.to_numpy(dtype=float))[0, 1])


def read_metadata(path: Path | None, n: int) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame({"row_index": np.arange(n, dtype=int)})
    meta = pd.read_csv(path)
    if "row_index" not in meta.columns:
        meta = meta.copy()
        meta.insert(0, "row_index", np.arange(len(meta), dtype=int))
    if len(meta) != n:
        raise ValueError(f"metadata row count differs from predictions: {len(meta)} != {n}")
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description="Assess MPro-URV DCML predictions by errors and pIC50 ranges.")
    p.add_argument("--predictions-csv", type=Path, required=True)
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=25)
    args = p.parse_args()

    pred = pd.read_csv(args.predictions_csv)
    required = {"sample_id", "true_affinity", "predicted_affinity"}
    missing = sorted(required - set(pred.columns))
    if missing:
        raise ValueError(f"Missing columns in predictions CSV: {missing}")

    pred = pred.copy()
    pred["row_index"] = np.arange(len(pred), dtype=int)
    pred["error"] = pred["predicted_affinity"] - pred["true_affinity"]
    pred["abs_error"] = pred["error"].abs()

    meta = read_metadata(args.metadata_csv, len(pred))
    merged = pd.concat([meta.reset_index(drop=True), pred.drop(columns=["row_index"]).reset_index(drop=True)], axis=1)

    bins = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    labels = ["4-5", "5-6", "6-7", "7-8", "8-9"]
    merged["pIC50_bin"] = pd.cut(merged["true_affinity"], bins=bins, labels=labels, include_lowest=True)

    rows = []
    for name, g in merged.groupby("pIC50_bin", dropna=False, observed=False):
        if len(g) == 0:
            continue
        rows.append({
            "pIC50_bin": str(name),
            "n": int(len(g)),
            "true_mean": float(g["true_affinity"].mean()),
            "pred_mean": float(g["predicted_affinity"].mean()),
            "bias_pred_minus_true": float(g["error"].mean()),
            "RMSE": rmse(g["error"]),
            "MAE": float(g["abs_error"].mean()),
            "Pearson": pearson(g["true_affinity"], g["predicted_affinity"]),
        })
    by_bin = pd.DataFrame(rows)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "predictions_csv": str(args.predictions_csv),
        "metadata_csv": str(args.metadata_csv) if args.metadata_csv else None,
        "n_samples": int(len(merged)),
        "metrics": {
            "RMSE": rmse(merged["error"]),
            "MAE": float(merged["abs_error"].mean()),
            "Pearson": pearson(merged["true_affinity"], merged["predicted_affinity"]),
            "bias_pred_minus_true": float(merged["error"].mean()),
        },
        "true_affinity_range": [float(merged["true_affinity"].min()), float(merged["true_affinity"].max())],
        "predicted_affinity_range": [float(merged["predicted_affinity"].min()), float(merged["predicted_affinity"].max())],
        "true_affinity_std": float(merged["true_affinity"].std(ddof=1)),
        "predicted_affinity_std": float(merged["predicted_affinity"].std(ddof=1)),
        "top_n": int(args.top_n),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / "predictions_with_errors.csv", index=False)
    by_bin.to_csv(args.out_dir / "error_by_pic50_bin.csv", index=False)
    merged.sort_values("abs_error", ascending=False).head(args.top_n).to_csv(args.out_dir / "worst_abs_errors.csv", index=False)
    merged.sort_values("error", ascending=True).head(args.top_n).to_csv(args.out_dir / "worst_underpredictions.csv", index=False)
    merged.sort_values("error", ascending=False).head(args.top_n).to_csv(args.out_dir / "worst_overpredictions.csv", index=False)
    (args.out_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote diagnostics to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
