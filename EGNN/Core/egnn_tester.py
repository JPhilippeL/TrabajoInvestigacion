"""
@file egnn_tester.py
@author Mohamed EL BOUKHIARI
@brief Testing and evaluation pipeline for the EGNN model.
@details
This file is adapted from 04_d_Predict_EGNN.py.
It exposes a callable test_model(...) function that evaluates the trained
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

from .egnn_model import EGNN


# ============================================================
# UTILS
# ============================================================

def load_split_txt(path: str):
    """
    @brief Load split indices from a text file.
    @param path Path to split file.
    @return Parsed split structure.
    """
    with open(path, "r") as f:
        return ast.literal_eval(f.read())


# ============================================================
# DATASET
# ============================================================

class URVGraphDataset(Dataset):
    """
    @brief Dataset wrapper for EGNN test graphs.
    @param graph_ids List of graph identifiers.
    @param graphs_dir Directory containing graph files.
    """

    def __init__(self, graph_ids, graphs_dir):
        self.graph_ids = graph_ids
        self.graphs_dir = graphs_dir

    def __len__(self):
        """
        @brief Return dataset size.
        @return Number of graphs.
        """
        return len(self.graph_ids)

    def __getitem__(self, idx):
        """
        @brief Load one graph by index.
        @param idx Graph index.
        @return PyG Data object.
        """
        pdb_id = self.graph_ids[idx]
        return torch.load(os.path.join(self.graphs_dir, f"{pdb_id}.pt"))


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, dataloader):
    """
    @brief Evaluate a trained model on one dataloader.
    @param model Trained EGNN model.
    @param dataloader Evaluation dataloader.
    @return Labels, predictions, and computed metrics according to original implementation.
    """

    model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for data in dataloader:

            data = data.to(DEVICE)

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

def plot_split_scatter(labels, preds, split_name, save_dir, rmse, pearson):
    """
    @brief Generate the scatter plot for one split.
    @param labels Ground truth labels.
    @param preds Model predictions.
    @param split_name Split identifier.
    @param save_dir Output directory.
    @param rmse RMSE value for display.
    @param pearson Pearson value for display.
    @return None
    """

    mask = (preds >= global_min) & (preds <= global_max)
    n_out = (~mask).sum()

    plt.figure(figsize=(5,5))
    plt.scatter(labels[mask], preds[mask], alpha=0.6)

    plt.xlim(global_min, global_max)
    plt.ylim(global_min, global_max)

    plt.xlabel("Valor real (pIC50)")
    plt.ylabel("Valor predicho (pIC50)")

    plt.title(f"EGNN – {split_name}\nRMSE = {rmse:.3f} | Pearson = {pearson:.3f}")

    plt.plot([global_min, global_max], [global_min, global_max], "r--")

    if n_out > 0:
        plt.figtext(
            0.5, 0.01,
            f"{n_out} non visible for being out of the domain",
            ha="center", fontsize=8
        )

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"scatter_{split_name}.png")

    plt.savefig(save_path)
    plt.close()

    print(f"Scatter {split_name} guardado en: {save_path}")


def plot_global_scatter(all_labels, all_preds, save_path, mean_rmse, mean_pearson):
    """
    @brief Generate the global scatter plot across all splits.
    @param all_labels Concatenated labels.
    @param all_preds Concatenated predictions.
    @param save_path Output image path.
    @param mean_rmse Mean RMSE over splits.
    @param mean_pearson Mean Pearson over splits.
    @return None
    """

    mask = (all_preds >= global_min) & (all_preds <= global_max)
    n_out = (~mask).sum()

    plt.figure(figsize=(6,6))

    plt.scatter(all_labels[mask], all_preds[mask], alpha=0.6)

    plt.xlim(global_min, global_max)
    plt.ylim(global_min, global_max)

    plt.xlabel("Valor real (pIC50)")
    plt.ylabel("Valor predicho (pIC50)")

    plt.title(f"EGNN - GLOBAL\nRMSE = {mean_rmse:.3f} | Pearson = {mean_pearson:.3f}")

    plt.plot([global_min, global_max], [global_min, global_max], "r--")

    if n_out > 0:
        plt.figtext(
            0.5, 0.01,
            f"{n_out} non visible for being out of the domain",
            ha="center", fontsize=8
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
    @brief Evaluate EGNN trained models over the predefined test splits.
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
            shuffle=False
        )

        split_model_dir = os.path.join(models_dir, f"split_{split_idx:02d}")
        model_path = os.path.join(split_model_dir, "best_model.pt")

        if not os.path.exists(model_path):
            print(f"[WARNING] Falta best_model.pt en {split_model_dir}")
            continue

        model = EGNN(hidden_dim=hidden_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        rmse, pearson, spearman, labels, preds = evaluate(model, test_loader)

        print(f"RMSE: {rmse:.4f}")
        print(f"Pearson: {pearson:.4f}")
        print(f"Spearman: {spearman:.4f}")

        plot_split_scatter(labels, preds, split_name, results_dir, rmse, pearson)

        all_results.append({
            "Split": split_idx,
            "RMSE": rmse,
            "Pearson": pearson,
            "Spearman": spearman
        })

        all_labels_global.append(labels)
        all_preds_global.append(preds)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(results_dir, "metrics_per_split.csv"),
        index=False
    )

    mean_row = results_df[["RMSE", "Pearson", "Spearman"]].mean()
    std_row = results_df[["RMSE", "Pearson", "Spearman"]].std()

    summary_df = pd.DataFrame({
        "Metric": ["RMSE", "Pearson", "Spearman"],
        "Mean": mean_row.values,
        "Std": std_row.values
    })

    summary_df.to_csv(
        os.path.join(results_dir, "metrics_summary.csv"),
        index=False
    )

    print("\n=== RESULTADOS FINALES ===")
    print(summary_df)

    all_labels_global = np.concatenate(all_labels_global)
    all_preds_global = np.concatenate(all_preds_global)

    np.save(os.path.join(results_dir, "egnn_labels.npy"), all_labels_global)
    np.save(os.path.join(results_dir, "egnn_preds.npy"), all_preds_global)

    scatter_path = os.path.join(results_dir, "scatter_global.png")

    plot_global_scatter(
        all_labels_global,
        all_preds_global,
        scatter_path,
        mean_row["RMSE"],
        mean_row["Pearson"]
    )

    print(f"\nScatter global guardado en: {scatter_path}")

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

    csv_path = os.path.join(results_root, "resumen_metricas_modelos.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return csv_path, all_metrics


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    MODULE_ROOT = os.path.dirname(PROJECT_ROOT)

    test_model(
        graphs_dir=os.path.join(MODULE_ROOT, "Graphs_EGNN"),
        dataset_dir=os.path.join(MODULE_ROOT, "MPro-URV_Version2"),
        models_dir=os.path.join(MODULE_ROOT, "Models_EGNN"),
        results_dir=os.path.join(MODULE_ROOT, "Results_EGNN"),
    )
