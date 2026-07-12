"""
@file egnn_tester.py
@author Mohamed EL BOUKHIARI
@brief Testing and evaluation pipeline for the EGNN model.
@details
This file is adapted from 04_d_Predict_EGNN.py.
It exposes callable functions that evaluate trained models and store metrics
and plots.
"""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from .egnn_model import EGNN


CHECKPOINT_FILENAMES = ("best_model.pt", "model.pt", "checkpoint.pt")


def load_split_txt(path: str):
    """
    @brief Load split indices from a text file.
    @param path Path to split file.
    @return Parsed split structure.
    """
    with open(path, "r", encoding="utf-8") as f:
        return ast.literal_eval(f.read())


def resolve_device(device: str | None) -> str:
    """
    @brief Resolve the requested device into a valid torch device string.
    @param device Requested device. Use None or "auto" for automatic selection.
    @return Device string.
    """
    if device is None or device == "" or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA was requested but is not available. Falling back to CPU.")
        return "cpu"

    return device


def safe_torch_load(path: str, map_location: str | None = None) -> Any:
    """
    @brief Load PyTorch objects while remaining compatible with PyTorch versions
           that introduced weights_only=True as a safer default.
    @param path File path.
    @param map_location Optional map_location.
    @return Loaded object.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class URVGraphDataset(Dataset):
    """
    @brief Dataset wrapper for EGNN test graphs.
    @param graph_ids List of graph identifiers.
    @param graphs_dir Directory containing graph files.
    """

    def __init__(self, graph_ids: list[str], graphs_dir: str):
        self.graph_ids = graph_ids
        self.graphs_dir = graphs_dir

    def __len__(self) -> int:
        return len(self.graph_ids)

    def __getitem__(self, idx: int):
        pdb_id = self.graph_ids[idx]
        return safe_torch_load(os.path.join(self.graphs_dir, f"{pdb_id}.pt"))


def evaluate(model: EGNN, dataloader: DataLoader, device: str):
    """
    @brief Evaluate a trained model on one dataloader.
    @param model Trained EGNN model.
    @param dataloader Evaluation dataloader.
    @param device Computation device.
    @return RMSE, Pearson, Spearman, labels, predictions.
    """
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            pred = model(data)

            preds.append(pred.cpu().numpy())
            labels.append(data.y.view(-1).cpu().numpy())

    if not preds:
        empty = np.array([])
        return float("nan"), float("nan"), float("nan"), empty, empty

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    rmse = float(np.sqrt(mean_squared_error(labels, preds)))
    pearson = float(np.corrcoef(labels, preds)[0, 1]) if len(labels) > 1 else float("nan")
    spearman = float(spearmanr(labels, preds)[0]) if len(labels) > 1 else float("nan")

    return rmse, pearson, spearman, labels, preds


def compute_axis_limits(labels: np.ndarray, margin_ratio: float = 0.05) -> tuple[float, float]:
    """
    @brief Compute plot axis limits from ground-truth labels.
    @param labels Ground-truth labels.
    @param margin_ratio Margin ratio around min/max values.
    @return Tuple (axis_min, axis_max).
    """
    finite_labels = labels[np.isfinite(labels)]

    if finite_labels.size == 0:
        return 0.0, 1.0

    axis_min = float(np.min(finite_labels))
    axis_max = float(np.max(finite_labels))

    if axis_min == axis_max:
        return axis_min - 0.5, axis_max + 0.5

    margin = (axis_max - axis_min) * margin_ratio
    return axis_min - margin, axis_max + margin


def plot_split_scatter(
    labels: np.ndarray,
    preds: np.ndarray,
    split_name: str,
    save_dir: str,
    rmse: float,
    pearson: float,
):
    """
    @brief Generate the scatter plot for one split.
    @param labels Ground-truth labels.
    @param preds Model predictions.
    @param split_name Split identifier.
    @param save_dir Output directory.
    @param rmse RMSE value for display.
    @param pearson Pearson value for display.
    @return None.
    """
    axis_min, axis_max = compute_axis_limits(labels)
    mask = (preds >= axis_min) & (preds <= axis_max)
    n_out = int((~mask).sum())

    plt.figure(figsize=(5, 5))
    plt.scatter(labels[mask], preds[mask], alpha=0.6)

    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)

    plt.xlabel("True value (pIC50)")
    plt.ylabel("Predicted value (pIC50)")
    plt.title(f"EGNN - {split_name}\nRMSE = {rmse:.3f} | Pearson = {pearson:.3f}")
    plt.plot([axis_min, axis_max], [axis_min, axis_max], "r--")

    if n_out > 0:
        plt.figtext(
            0.5,
            0.01,
            f"{n_out} predictions are outside the visible axis range.",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"scatter_{split_name.replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Scatter plot for {split_name} saved to: {save_path}")


def plot_global_scatter(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    save_path: str,
    mean_rmse: float,
    mean_pearson: float,
):
    """
    @brief Generate the global scatter plot across all splits.
    @param all_labels Concatenated labels.
    @param all_preds Concatenated predictions.
    @param save_path Output image path.
    @param mean_rmse Mean RMSE over splits.
    @param mean_pearson Mean Pearson over splits.
    @return None.
    """
    axis_min, axis_max = compute_axis_limits(all_labels)
    mask = (all_preds >= axis_min) & (all_preds <= axis_max)
    n_out = int((~mask).sum())

    plt.figure(figsize=(6, 6))
    plt.scatter(all_labels[mask], all_preds[mask], alpha=0.6)

    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)

    plt.xlabel("True value (pIC50)")
    plt.ylabel("Predicted value (pIC50)")
    plt.title(f"EGNN - GLOBAL\nRMSE = {mean_rmse:.3f} | Pearson = {mean_pearson:.3f}")
    plt.plot([axis_min, axis_max], [axis_min, axis_max], "r--")

    if n_out > 0:
        plt.figtext(
            0.5,
            0.01,
            f"{n_out} predictions are outside the visible axis range.",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _format_split_dir(split_idx: int) -> str:
    return f"split_{split_idx:02d}"


def _find_direct_checkpoint(path: str) -> str | None:
    if os.path.isfile(path):
        return path

    for filename in CHECKPOINT_FILENAMES:
        candidate = os.path.join(path, filename)
        if os.path.isfile(candidate):
            return candidate

    pt_files = sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.endswith(".pt") and os.path.isfile(os.path.join(path, name))
    )
    return pt_files[0] if len(pt_files) == 1 else None


def _detect_split_checkpoints(run_root: str, expected_count: int) -> tuple[dict[int, str], list[str]]:
    found: dict[int, str] = {}

    for name in sorted(os.listdir(run_root)):
        direct_match = re.search(r"split[_-]?(\d+)", name)
        direct_path = os.path.join(run_root, name)
        if direct_match and name.endswith(".pt") and os.path.isfile(direct_path):
            found[int(direct_match.group(1))] = direct_path
            continue

        match = re.fullmatch(r"split_(\d+)", name)
        if not match:
            continue

        split_dir = os.path.join(run_root, name)
        if not os.path.isdir(split_dir):
            continue

        checkpoint = _find_direct_checkpoint(split_dir)
        if checkpoint:
            found[int(match.group(1))] = checkpoint

    missing = []
    if found:
        for split_idx in range(expected_count):
            if split_idx not in found:
                missing.append(os.path.join(run_root, _format_split_dir(split_idx), "best_model.pt"))

    return found, missing


def _write_metrics(metrics: dict[str, Any], save_dir: str) -> None:
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, "metrics.csv"), index=False)


def _write_predictions(graph_ids: list[str], labels: np.ndarray, preds: np.ndarray, save_dir: str) -> None:
    pd.DataFrame(
        {
            "graph_id": graph_ids,
            "label": labels.reshape(-1),
            "prediction": preds.reshape(-1),
        }
    ).to_csv(os.path.join(save_dir, "predictions.csv"), index=False)


def _evaluate_one_checkpoint(
    graphs_dir: str,
    test_splits: list[list[str]],
    checkpoint_path: str,
    output_dir: str,
    split_idx: int,
    batch_size: int,
    device: str,
    hidden_dim: int,
) -> dict[str, Any]:
    if split_idx < 0 or split_idx >= len(test_splits):
        raise ValueError(f"Split index {split_idx} is outside the available split range 0..{len(test_splits) - 1}.")

    split_name = f"Split {split_idx:02d}"
    print(f"[EGNN Evaluation] Split being evaluated: {split_name}")
    print(f"[EGNN Evaluation] Checkpoint found: {checkpoint_path}")

    os.makedirs(output_dir, exist_ok=True)
    test_ids = test_splits[split_idx]
    test_set = URVGraphDataset(test_ids, graphs_dir)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = EGNN(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(safe_torch_load(checkpoint_path, map_location=device))

    rmse, pearson, spearman, labels, preds = evaluate(model, test_loader, device)
    metrics = {
        "Split": split_idx,
        "RMSE": rmse,
        "Pearson": pearson,
        "Spearman": spearman,
        "checkpoint": checkpoint_path,
    }

    _write_metrics(metrics, output_dir)
    _write_predictions(test_ids, labels, preds, output_dir)
    plot_split_scatter(labels, preds, split_name, output_dir, rmse, pearson)

    print(f"[EGNN Evaluation] Metrics for {split_name}: RMSE={rmse:.4f}, Pearson={pearson:.4f}, Spearman={spearman:.4f}")
    print(f"[EGNN Evaluation] Output files written in: {output_dir}")

    return {
        **metrics,
        "labels": labels,
        "predictions": preds,
        "graph_ids": test_ids,
        "results_dir": output_dir,
    }


def evaluate_checkpoint_or_run(
    graphs_dir: str,
    test_split_file: str,
    checkpoint_or_run: str,
    results_dir: str,
    batch_size: int = 4,
    device: str | None = None,
    hidden_dim: int = 64,
    split_index: int = 0,
    evaluation_scope: str = "auto",
) -> dict[str, Any]:
    """
    @brief Evaluate one EGNN checkpoint or all detected split checkpoints in a trained run.
    """
    print("[EGNN Evaluation] Start evaluation.")
    device = resolve_device(device)
    scope = (evaluation_scope or "auto").strip().lower().replace(" ", "_")
    test_splits = load_split_txt(test_split_file)
    os.makedirs(results_dir, exist_ok=True)

    if not checkpoint_or_run or not os.path.exists(checkpoint_or_run):
        raise FileNotFoundError(f"No checkpoint or trained run was found at: {checkpoint_or_run}")

    if scope not in {"auto", "single_checkpoint", "all_detected_splits"}:
        raise ValueError(f"Unknown evaluation scope: {evaluation_scope}")

    split_checkpoints: dict[int, str] = {}
    missing_split_checkpoints: list[str] = []
    direct_checkpoint = _find_direct_checkpoint(checkpoint_or_run)

    if os.path.isdir(checkpoint_or_run):
        split_checkpoints, missing_split_checkpoints = _detect_split_checkpoints(checkpoint_or_run, len(test_splits))

    if scope == "all_detected_splits" or (scope == "auto" and split_checkpoints):
        if not split_checkpoints:
            raise FileNotFoundError(
                f"No split checkpoints found in {checkpoint_or_run}. "
                "Expected split_00/best_model.pt style folders."
            )

        print("[EGNN Evaluation] Detected mode: all detected splits.")
        print(f"[EGNN Evaluation] Checkpoint(s) found: {len(split_checkpoints)}")
        if missing_split_checkpoints:
            print("[EGNN Evaluation] Missing expected split checkpoint(s):")
            for path in missing_split_checkpoints:
                print(f"  - {path}")

        split_results = []
        all_labels_global = []
        all_preds_global = []
        combined_predictions = []

        for split_idx, checkpoint_path in sorted(split_checkpoints.items()):
            split_output_dir = os.path.join(results_dir, _format_split_dir(split_idx))
            result = _evaluate_one_checkpoint(
                graphs_dir=graphs_dir,
                test_splits=test_splits,
                checkpoint_path=checkpoint_path,
                output_dir=split_output_dir,
                split_idx=split_idx,
                batch_size=batch_size,
                device=device,
                hidden_dim=hidden_dim,
            )
            split_results.append({k: result[k] for k in ("Split", "RMSE", "Pearson", "Spearman", "checkpoint")})
            all_labels_global.append(result["labels"])
            all_preds_global.append(result["predictions"])
            combined_predictions.append(
                pd.DataFrame(
                    {
                        "split": split_idx,
                        "graph_id": result["graph_ids"],
                        "label": result["labels"].reshape(-1),
                        "prediction": result["predictions"].reshape(-1),
                    }
                )
            )

        results_df = pd.DataFrame(split_results)
        results_df.to_csv(os.path.join(results_dir, "metrics_per_split.csv"), index=False)

        mean_row = results_df[["RMSE", "Pearson", "Spearman"]].mean()
        std_row = results_df[["RMSE", "Pearson", "Spearman"]].std()
        summary_df = pd.DataFrame(
            {
                "Metric": ["RMSE", "Pearson", "Spearman"],
                "Mean": mean_row.values,
                "Std": std_row.values,
            }
        )
        summary_df.to_csv(os.path.join(results_dir, "metrics_summary.csv"), index=False)
        pd.concat(combined_predictions, ignore_index=True).to_csv(
            os.path.join(results_dir, "predictions.csv"),
            index=False,
        )

        all_labels = np.concatenate(all_labels_global)
        all_preds = np.concatenate(all_preds_global)
        np.save(os.path.join(results_dir, "egnn_labels.npy"), all_labels)
        np.save(os.path.join(results_dir, "egnn_preds.npy"), all_preds)
        plot_global_scatter(
            all_labels,
            all_preds,
            os.path.join(results_dir, "scatter_global.png"),
            float(mean_row["RMSE"]),
            float(mean_row["Pearson"]),
        )

        print(f"[EGNN Evaluation] Aggregated output files written in: {results_dir}")
        print("[EGNN Evaluation] Done.")
        return {
            "mode": "all_detected_splits",
            "RMSE": float(mean_row["RMSE"]),
            "Pearson": float(mean_row["Pearson"]),
            "Spearman": float(mean_row["Spearman"]),
            "missing_checkpoints": missing_split_checkpoints,
            "results_dir": results_dir,
        }

    if not direct_checkpoint:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_or_run}. "
            f"Expected one of: {', '.join(CHECKPOINT_FILENAMES)}"
        )

    print("[EGNN Evaluation] Detected mode: single checkpoint.")
    result = _evaluate_one_checkpoint(
        graphs_dir=graphs_dir,
        test_splits=test_splits,
        checkpoint_path=direct_checkpoint,
        output_dir=results_dir,
        split_idx=split_index,
        batch_size=batch_size,
        device=device,
        hidden_dim=hidden_dim,
    )
    print("[EGNN Evaluation] Done.")
    return {
        "mode": "single_checkpoint",
        "RMSE": float(result["RMSE"]),
        "Pearson": float(result["Pearson"]),
        "Spearman": float(result["Spearman"]),
        "results_dir": results_dir,
    }


def test_model(
    graphs_dir: str,
    test_split_file: str,
    models_dir: str,
    results_dir: str,
    batch_size: int = 4,
    device: str | None = None,
    hidden_dim: int = 64,
):
    return evaluate_checkpoint_or_run(
        graphs_dir=graphs_dir,
        test_split_file=test_split_file,
        checkpoint_or_run=models_dir,
        results_dir=results_dir,
        batch_size=batch_size,
        device=device,
        hidden_dim=hidden_dim,
        evaluation_scope="auto",
    )


def test_all_models_in_folder(
    graphs_dir: str,
    test_split_file: str,
    models_root: str,
    results_root: str,
    batch_size: int = 4,
    device: str | None = None,
    hidden_dim: int = 64,
):
    """
    @brief Evaluate all experiment folders contained in a root directory.
    @param graphs_dir Directory containing generated graphs.
    @param test_split_file Path to test_index_folder.txt.
    @param models_root Root directory containing multiple experiment folders.
    @param results_root Root directory where evaluation outputs will be stored.
    @param batch_size Batch size.
    @param device Device to use.
    @param hidden_dim Hidden dimension.
    @return Tuple (csv_path, all_metrics_dict).
    """
    os.makedirs(results_root, exist_ok=True)

    all_metrics = {}
    rows = []

    subdirs = [
        os.path.join(models_root, d)
        for d in sorted(os.listdir(models_root))
        if os.path.isdir(os.path.join(models_root, d))
    ]

    for subdir in subdirs:
        model_name = os.path.basename(subdir)
        result_dir = os.path.join(results_root, model_name)

        try:
            metrics = test_model(
                graphs_dir=graphs_dir,
                test_split_file=test_split_file,
                models_dir=subdir,
                results_dir=result_dir,
                batch_size=batch_size,
                device=device,
                hidden_dim=hidden_dim,
            )
        except Exception as exc:
            print(f"[WARNING] Evaluation failed for {model_name}: {exc}")
            continue

        all_metrics[model_name] = metrics
        rows.append(
            {
                "Model": model_name,
                "RMSE": metrics["RMSE"],
                "Pearson": metrics["Pearson"],
                "Spearman": metrics["Spearman"],
            }
        )

    csv_path = os.path.join(results_root, "model_metrics_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return csv_path, all_metrics


if __name__ == "__main__":
    from EGNN.utils.constants import (
        DEFAULT_GRAPHS_DIR,
        DEFAULT_MODELS_DIR,
        DEFAULT_RESULTS_DIR,
        DEFAULT_TEST_SPLIT_FILE,
    )

    test_model(
        graphs_dir=DEFAULT_GRAPHS_DIR,
        test_split_file=DEFAULT_TEST_SPLIT_FILE,
        models_dir=DEFAULT_MODELS_DIR,
        results_dir=DEFAULT_RESULTS_DIR,
    )
