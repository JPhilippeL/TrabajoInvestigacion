"""
@file deepdta_evaluator.py
@brief Explicit DeepDTA checkpoint evaluation and prediction export.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from DeepDTA.Core.deepdta_audit import audit_dataset_splits, get_split_indices
from DeepDTA.Core.deepdta_trainer import RMSLoss, evaluate, get_dataset_paths, load_saved_model, prepare_batch, resolve_device
from DeepDTA.data import NumbersDataset


def _extra_metrics(predictions: list[float], targets: list[float]) -> dict[str, float | int]:
    pred = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(targets, dtype=np.float64)
    mse = float(np.mean((pred - true) ** 2)) if pred.size else float("nan")
    mae = float(np.mean(np.abs(pred - true))) if pred.size else float("nan")
    rmse = float(np.sqrt(mse)) if pred.size else float("nan")
    pearson = float(np.corrcoef(pred, true)[0, 1]) if pred.size > 1 and np.std(pred) and np.std(true) else float("nan")
    result: dict[str, float | int] = {"RMSE": rmse, "MSE": mse, "MAE": mae, "Pearson": pearson, "n_samples": int(pred.size)}
    try:
        from scipy.stats import spearmanr
        result["Spearman"] = float(spearmanr(true, pred).correlation)
    except Exception:
        result["Spearman"] = float("nan")
    try:
        from sklearn.metrics import r2_score
        result["R2"] = float(r2_score(true, pred)) if pred.size > 1 else float("nan")
    except Exception:
        result["R2"] = float("nan")
    return result


def _write_yaml_or_json(data: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        if yaml is not None:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
        else:
            json.dump(data, file, indent=4, default=str)


def _plot_scatter(rows: list[dict[str, Any]], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    y_true = [row["y_true"] for row in rows]
    y_pred = [row["y_pred"] for row in rows]
    if not y_true:
        return
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.75)
    low = min(min(y_true), min(y_pred))
    high = max(max(y_true), max(y_pred))
    plt.plot([low, high], [low, high], "k--", linewidth=1)
    plt.xlabel("True pIC50")
    plt.ylabel("Predicted pIC50")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset_name: str = "mpro_urv",
    output_dir: str | None = None,
    device: str = "auto",
    fold_index: int = 0,
    use_dataset_folds: bool = True,
    split: str = "test",
    batch_size: int = 4,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    if split not in {"train", "valid", "test", "all"}:
        raise ValueError("split must be one of: train, valid, test, all")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "deepdta_evaluation", "runs", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    audit = audit_dataset_splits(dataset_name, fold_index, use_dataset_folds, val_split, test_split, seed)
    audit_path = os.path.join(output_dir, "split_audit.json")
    with open(audit_path, "w", encoding="utf-8") as file:
        json.dump(audit, file, indent=4, default=str)

    ligand_path, protein_path, affinity_path = get_dataset_paths(dataset_name)
    dataset = NumbersDataset(ligand_path, protein_path, affinity_path)
    splits, split_mode = get_split_indices(dataset_name, fold_index, use_dataset_folds, val_split, test_split, seed)
    selected_splits = ["train", "valid", "test"] if split == "all" else [split]
    torch_device = resolve_device(device)
    model = load_saved_model(checkpoint_path, torch_device)
    criterion = RMSLoss()

    all_rows: list[dict[str, Any]] = []
    metrics_by_split: dict[str, Any] = {}
    ligand_ids = list(__import__("json").load(open(ligand_path, encoding="utf-8")).keys())

    for split_name in selected_splits:
        indices = splits[split_name]
        loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
        base_metrics = evaluate(model, loader, criterion, torch_device)
        rows: list[dict[str, Any]] = []
        cursor = 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                ligand, protein, target = prepare_batch(batch, torch_device)
                output = model(ligand, protein)
                y_pred = output.detach().cpu().numpy().reshape(-1).tolist()
                y_true = target.detach().cpu().numpy().reshape(-1).tolist()
                for pred, true in zip(y_pred, y_true):
                    sample_index = int(indices[cursor])
                    ligand_id = ligand_ids[sample_index] if sample_index < len(ligand_ids) else ""
                    row = {
                        "sample_index": sample_index,
                        "sample_id": ligand_id,
                        "ligand_id": ligand_id,
                        "protein_id": "Mpro",
                        "split": split_name,
                        "fold_index": fold_index if split_mode == "dataset_folds" else "",
                        "y_true": float(true),
                        "y_pred": float(pred),
                        "abs_error": float(abs(pred - true)),
                    }
                    rows.append(row)
                    all_rows.append(row)
                    cursor += 1
        metrics = _extra_metrics([r["y_pred"] for r in rows], [r["y_true"] for r in rows])
        metrics["loss"] = base_metrics.get("loss")
        metrics_by_split[split_name] = metrics
        _plot_scatter(rows, os.path.join(output_dir, f"scatter_{split_name}.png"))

    predictions_path = os.path.join(output_dir, "predictions.csv")
    timestamped_predictions_path = os.path.join(output_dir, f"predictions_deepdta_{timestamp}.csv")
    fieldnames = ["sample_index", "sample_id", "ligand_id", "protein_id", "split", "fold_index", "y_true", "y_pred", "abs_error"]
    with open(predictions_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    shutil.copy2(predictions_path, timestamped_predictions_path)

    run_config = {
        "model": "DeepDTA",
        "dataset": dataset_name,
        "checkpoint_path": checkpoint_path,
        "device": str(torch_device),
        "split": split,
        "split_mode": split_mode,
        "fold_index": fold_index if split_mode == "dataset_folds" else None,
        "batch_size": batch_size,
        "warnings": audit.get("warnings", []),
    }
    metrics_payload = {"model": "DeepDTA", "metrics_by_split": metrics_by_split, "warnings": audit.get("warnings", [])}
    metrics_path = os.path.join(output_dir, "metrics.json")
    timestamped_metrics_path = os.path.join(output_dir, f"metrics_deepdta_{timestamp}.json")
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=4, default=str)
    shutil.copy2(metrics_path, timestamped_metrics_path)
    _write_yaml_or_json(run_config, os.path.join(output_dir, "run_config.yaml"))

    return {
        "status": "success",
        "dataset": dataset_name,
        "checkpoint_path": checkpoint_path,
        "output_dir": output_dir,
        "predictions_csv": predictions_path,
        "timestamped_predictions_csv": timestamped_predictions_path,
        "metrics_json": metrics_path,
        "timestamped_metrics_json": timestamped_metrics_path,
        "split_audit_json": audit_path,
        "run_config_yaml": os.path.join(output_dir, "run_config.yaml"),
        "metrics": metrics_by_split,
        "warnings": audit.get("warnings", []),
    }
