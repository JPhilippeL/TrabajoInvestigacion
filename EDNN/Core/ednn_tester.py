"""
@file ednn_tester.py
@author Mohamed EL BOUKHIARI
@brief Testing and evaluation pipeline for the EDNN model.
@details
This file exposes a callable test_model(...) function that evaluates trained
models and stores the resulting metrics and plots.
"""

from __future__ import annotations

import os
import ast
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

from .ednn_model import EDNN


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

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    rmse = np.sqrt(mean_squared_error(labels, preds))
    pearson = np.corrcoef(labels, preds)[0, 1]
    spearman = spearmanr(labels, preds)[0]

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
    """
    @brief Evaluate EDNN trained models over the predefined test splits.
    @param graphs_dir Directory containing generated graphs.
    @param test_split_file Path to test_index_folder.txt.
    @param models_dir Directory containing split subdirectories with best_model.pt.
    @param results_dir Directory where evaluation outputs will be stored.
    @param batch_size Batch size for evaluation.
    @param device Device to use.
    @param hidden_dim Hidden dimension used to instantiate the model.
    @return Dictionary with summary metrics.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    all_labels_global = []
    all_preds_global = []

    test_splits = load_split_txt(test_split_file)

    for split_idx in range(5):
        split_name = f"Split {split_idx:02d}"

        print("\n==============================")
        print(f"        {split_name}")
        print("==============================")

        test_ids = test_splits[split_idx]
        test_set = URVGraphDataset(test_ids, graphs_dir)

        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
        )

        split_model_dir = os.path.join(models_dir, f"split_{split_idx:02d}")
        model_path = os.path.join(split_model_dir, "best_model.pt")

        if not os.path.exists(model_path):
            print(f"[WARNING] Missing best_model.pt in {split_model_dir}")
            continue

        model = EDNN(hidden_dim=hidden_dim).to(device)
        model.load_state_dict(load_state_dict(model_path, device))

        rmse, pearson, spearman, labels, preds = evaluate(model, test_loader, device)
        axis_min, axis_max = compute_axis_limits(labels)

        print(f"RMSE: {rmse:.4f}")
        print(f"Pearson: {pearson:.4f}")
        print(f"Spearman: {spearman:.4f}")

        plot_split_scatter(labels, preds, split_name, results_dir, rmse, pearson, axis_min, axis_max)

        all_results.append({
            "Split": split_idx,
            "RMSE": rmse,
            "Pearson": pearson,
            "Spearman": spearman,
        })

        all_labels_global.append(labels)
        all_preds_global.append(preds)

    if not all_results:
        raise RuntimeError("No EDNN model was evaluated. Check models_dir and split folders.")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(results_dir, "metrics_per_split.csv"),
        index=False,
    )

    mean_row = results_df[["RMSE", "Pearson", "Spearman"]].mean()
    std_row = results_df[["RMSE", "Pearson", "Spearman"]].std()

    summary_df = pd.DataFrame({
        "Metric": ["RMSE", "Pearson", "Spearman"],
        "Mean": mean_row.values,
        "Std": std_row.values,
    })

    summary_df.to_csv(
        os.path.join(results_dir, "metrics_summary.csv"),
        index=False,
    )

    print("\n=== FINAL RESULTS ===")
    print(summary_df)

    all_labels_global = np.concatenate(all_labels_global)
    all_preds_global = np.concatenate(all_preds_global)

    np.save(os.path.join(results_dir, "ednn_labels.npy"), all_labels_global)
    np.save(os.path.join(results_dir, "ednn_preds.npy"), all_preds_global)

    axis_min, axis_max = compute_axis_limits(all_labels_global)
    scatter_path = os.path.join(results_dir, "scatter_global.png")

    plot_global_scatter(
        all_labels_global,
        all_preds_global,
        scatter_path,
        mean_row["RMSE"],
        mean_row["Pearson"],
        axis_min,
        axis_max,
    )

    print(f"\nGlobal scatter saved in: {scatter_path}")

    return {
        "RMSE": mean_row["RMSE"],
        "Pearson": mean_row["Pearson"],
        "Spearman": mean_row["Spearman"],
        "results_dir": results_dir,
    }


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
