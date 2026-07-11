"""
@file ednn_tester.py
@author Mohamed EL BOUKHIARI
@brief Testing and evaluation pipeline for the EDNN model.
@details
This file exposes a callable test_model(...) function that evaluates trained
models and stores the resulting metrics and plots.
"""

from __future__ import annotations

import ast
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

from .ednn_model import EDNN


CHECKPOINT_FILENAMES = ("best_model.pt", "model.pt", "checkpoint.pt")


# ============================================================
# UTILS
# ============================================================

def load_split_txt(path: str):
    """
    @brief Load split indices from a text file.
    @param path Path to split file.
    @return Parsed split structure.
    """
    with open(path, "r", encoding="utf-8") as f:
        return ast.literal_eval(f.read())


def safe_torch_load(path: str):
    """
    @brief Load PyTorch objects while staying compatible with old/new torch versions.
    """
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def resolve_device(device: str | None) -> str:
    if device is None or device == "" or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA was requested but is not available. Falling back to CPU.")
        return "cpu"

    return device


def load_state_dict(path: str, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def compute_axis_limits(labels: np.ndarray) -> tuple[float, float]:
    """
    @brief Compute stable plotting bounds from ground-truth pIC50 labels.
    """
    lower = float(np.floor(np.min(labels) * 10.0) / 10.0)
    upper = float(np.ceil(np.max(labels) * 10.0) / 10.0)

    if lower == upper:
        lower -= 0.5
        upper += 0.5

    return lower, upper


# ============================================================
# DATASET
# ============================================================

class URVGraphDataset(Dataset):
    """
    @brief Dataset wrapper for EDNN test graphs.
    @param graph_ids List of graph identifiers.
    @param graphs_dir Directory containing graph files.
    """

    def __init__(self, graph_ids, graphs_dir):
        self.graph_ids = graph_ids
        self.graphs_dir = graphs_dir

    def __len__(self):
        return len(self.graph_ids)

    def __getitem__(self, idx):
        pdb_id = self.graph_ids[idx]
        return safe_torch_load(os.path.join(self.graphs_dir, f"{pdb_id}.pt"))


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, dataloader, device):
    """
    @brief Evaluate a trained model on one dataloader.
    @param model Trained EDNN model.
    @param dataloader Evaluation dataloader.
    @param device Device used for inference.
    @return Labels, predictions, and computed metrics.
    """
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            pred = model(data)

            preds.append(pred.cpu().numpy())
            labels.append(data.y.cpu().numpy())

    if not preds:
        empty = np.array([])
        return float("nan"), float("nan"), float("nan"), empty, empty

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    rmse = float(np.sqrt(mean_squared_error(labels, preds)))
    pearson = float(np.corrcoef(labels, preds)[0, 1]) if len(labels) > 1 else float("nan")
    spearman = float(spearmanr(labels, preds)[0]) if len(labels) > 1 else float("nan")

    return rmse, pearson, spearman, labels, preds


# ============================================================
# PLOTS
# ============================================================

def plot_split_scatter(labels, preds, split_name, save_dir, rmse, pearson, axis_min, axis_max):
    """
    @brief Generate the scatter plot for one split.
    """
    mask = (preds >= axis_min) & (preds <= axis_max)
    n_out = int((~mask).sum())

    plt.figure(figsize=(5, 5))
    plt.scatter(labels[mask], preds[mask], alpha=0.6)

    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)

    plt.xlabel("Real value (pIC50)")
    plt.ylabel("Predicted value (pIC50)")
    plt.title(f"EDNN - {split_name}\nRMSE = {rmse:.3f} | Pearson = {pearson:.3f}")

    plt.plot([axis_min, axis_max], [axis_min, axis_max], "r--")

    if n_out > 0:
        plt.figtext(
            0.5,
            0.01,
            f"{n_out} predictions outside plot domain",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"scatter_{split_name.replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Scatter {split_name} saved in: {save_path}")


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


def _write_metrics(metrics: dict, save_dir: str) -> None:
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
) -> dict:
    if split_idx < 0 or split_idx >= len(test_splits):
        raise ValueError(f"Split index {split_idx} is outside the available split range 0..{len(test_splits) - 1}.")

    split_name = f"Split {split_idx:02d}"
    print(f"[EDNN Evaluation] Split being evaluated: {split_name}")
    print(f"[EDNN Evaluation] Checkpoint found: {checkpoint_path}")

    os.makedirs(output_dir, exist_ok=True)
    test_ids = test_splits[split_idx]
    test_set = URVGraphDataset(test_ids, graphs_dir)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = EDNN(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(load_state_dict(checkpoint_path, device))

    rmse, pearson, spearman, labels, preds = evaluate(model, test_loader, device)
    axis_min, axis_max = compute_axis_limits(labels)
    metrics = {
        "Split": split_idx,
        "RMSE": rmse,
        "Pearson": pearson,
        "Spearman": spearman,
        "checkpoint": checkpoint_path,
    }

    _write_metrics(metrics, output_dir)
    _write_predictions(test_ids, labels, preds, output_dir)
    plot_split_scatter(labels, preds, split_name, output_dir, rmse, pearson, axis_min, axis_max)

    print(f"[EDNN Evaluation] Metrics for {split_name}: RMSE={rmse:.4f}, Pearson={pearson:.4f}, Spearman={spearman:.4f}")
    print(f"[EDNN Evaluation] Output files written in: {output_dir}")

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
) -> dict:
    """
    @brief Evaluate one EDNN checkpoint or all detected split checkpoints in a trained run.
    """
    print("[EDNN Evaluation] Start evaluation.")
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

        print("[EDNN Evaluation] Detected mode: all detected splits.")
        print(f"[EDNN Evaluation] Checkpoint(s) found: {len(split_checkpoints)}")
        if missing_split_checkpoints:
            print("[EDNN Evaluation] Missing expected split checkpoint(s):")
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
        np.save(os.path.join(results_dir, "ednn_labels.npy"), all_labels)
        np.save(os.path.join(results_dir, "ednn_preds.npy"), all_preds)
        axis_min, axis_max = compute_axis_limits(all_labels)
        plot_global_scatter(
            all_labels,
            all_preds,
            os.path.join(results_dir, "scatter_global.png"),
            float(mean_row["RMSE"]),
            float(mean_row["Pearson"]),
            axis_min,
            axis_max,
        )

        print(f"[EDNN Evaluation] Aggregated output files written in: {results_dir}")
        print("[EDNN Evaluation] Done.")
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

    print("[EDNN Evaluation] Detected mode: single checkpoint.")
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
    print("[EDNN Evaluation] Done.")
    return {
        "mode": "single_checkpoint",
        "RMSE": float(result["RMSE"]),
        "Pearson": float(result["Pearson"]),
        "Spearman": float(result["Spearman"]),
        "results_dir": results_dir,
    }

def plot_global_scatter(all_labels, all_preds, save_path, mean_rmse, mean_pearson, axis_min, axis_max):
    """
    @brief Generate the global scatter plot across all splits.
    """
    mask = (all_preds >= axis_min) & (all_preds <= axis_max)
    n_out = int((~mask).sum())

    plt.figure(figsize=(6, 6))
    plt.scatter(all_labels[mask], all_preds[mask], alpha=0.6)

    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)

    plt.xlabel("Real value (pIC50)")
    plt.ylabel("Predicted value (pIC50)")
    plt.title(f"EDNN - GLOBAL\nRMSE = {mean_rmse:.3f} | Pearson = {mean_pearson:.3f}")

    plt.plot([axis_min, axis_max], [axis_min, axis_max], "r--")

    if n_out > 0:
        plt.figtext(
            0.5,
            0.01,
            f"{n_out} predictions outside plot domain",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ============================================================
# TEST MODEL
# ============================================================

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
    @return Tuple (csv_path, all_metrics_dict).
    """
    os.makedirs(results_root, exist_ok=True)

    all_metrics = {}
    rows = []

    subdirs = [
        os.path.join(models_root, d)
        for d in os.listdir(models_root)
        if os.path.isdir(os.path.join(models_root, d))
    ]

    for subdir in subdirs:
        model_name = os.path.basename(subdir)
        result_dir = os.path.join(results_root, model_name)

        metrics = test_model(
            graphs_dir=graphs_dir,
            test_split_file=test_split_file,
            models_dir=subdir,
            results_dir=result_dir,
            batch_size=batch_size,
            device=device,
            hidden_dim=hidden_dim,
        )

        all_metrics[model_name] = metrics
        rows.append({
            "Model": model_name,
            "RMSE": metrics["RMSE"],
            "Pearson": metrics["Pearson"],
            "Spearman": metrics["Spearman"],
        })

    csv_path = os.path.join(results_root, "model_metrics_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return csv_path, all_metrics


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    MODULE_ROOT = os.path.dirname(PROJECT_ROOT)

    test_model(
        graphs_dir=os.path.join(MODULE_ROOT, "Graphs_EDNN"),
        test_split_file=os.path.join(MODULE_ROOT, "test_index_folder.txt"),
        models_dir=os.path.join(MODULE_ROOT, "Models_EDNN"),
        results_dir=os.path.join(MODULE_ROOT, "Results_EDNN"),
    )
