import os
import ast
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib

# don't remove that otherwise the thread will fail
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import torch.nn as nn

from GIGN_GUI.model.GIGN_model import GIGN

# =========================================================
# Utils
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_parameter_file(file):
    if not os.path.isfile(file):
        raise FileNotFoundError(f"Parameter file not found: {file}")

    params = dict.fromkeys(["node_dim", "hidden_dim", "drop_out", "batch_size"], None)

    with open(file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key not in params:
                continue

            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                print("Error parsing value for key '{}': {}".format(key, value))
                return None

            params[key] = value

    return params


def escala_global(file_path):
    labels = np.loadtxt(file_path, usecols=1)

    real_min = labels.min()
    real_max = labels.max()

    margin = 0.2
    global_min = real_min - margin
    global_max = real_max + margin

    print("Global axis limits:", global_min, global_max)

    np.save("global_axis.npy", [global_min, global_max])

    return global_min, global_max


def load_split_txt(path):
    with open(path, "r") as f:
        return ast.literal_eval(f.read())


# =========================================================
# Dataset
# =========================================================


class URVGraphDataset(Dataset):
    def __init__(self, graph_ids, graphs_dir):
        self.graph_ids = graph_ids
        self.graphs_dir = graphs_dir

    def __len__(self):
        return len(self.graph_ids)

    def __getitem__(self, idx):
        pdb_id = self.graph_ids[idx]
        return torch.load(
            os.path.join(self.graphs_dir, f"{pdb_id}.pt"), weights_only=False
        )


# =========================================================
# Evaluación
# =========================================================


def evaluate(model, dataloader):
    model.eval()
    preds, labels = [], []

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


# =========================================================
# Plots
# =========================================================


def plot_split_scatter(labels, preds, split_name, save_dir, rmse, pearson, global_min, global_max):
    mask = (preds >= global_min) & (preds <= global_max)
    n_out = (~mask).sum()

    plt.figure(figsize=(5, 5))
    plt.scatter(labels[mask], preds[mask], alpha=0.6)

    plt.xlim(global_min, global_max)
    plt.ylim(global_min, global_max)

    plt.xlabel("Valor real (pIC50)")
    plt.ylabel("Valor predicho (pIC50)")
    plt.title(
        f"GIGN – {split_name}\nRMSE = {rmse:.3f} | Pearson = {pearson:.3f}"
    )
    plt.plot([global_min, global_max], [global_min, global_max], "r--")

    # Nota al pie del gráfico
    if n_out > 0:
        plt.figtext(
            0.5,
            0.01,
            f"{n_out} non visible for being out of the domain",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"scatter_{split_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Scatter {split_name} guardado en: {save_path}")


def plot_global_scatter(
        all_labels, all_preds, save_path, mean_rmse, mean_pearson, global_min, global_max
):
    mask = (all_preds >= global_min) & (all_preds <= global_max)
    n_out = (~mask).sum()

    plt.figure(figsize=(6, 6))
    plt.scatter(all_labels[mask], all_preds[mask], alpha=0.6)

    plt.xlim(global_min, global_max)
    plt.ylim(global_min, global_max)

    plt.xlabel("Valor real (pIC50)")
    plt.ylabel("Valor predicho (pIC50)")
    plt.title(
        f"GIGN - GLOBAL\nRMSE = {mean_rmse:.3f} | Pearson = {mean_pearson:.3f}"
    )
    plt.plot([global_min, global_max], [global_min, global_max], "r--")

    if n_out > 0:
        plt.figtext(
            0.5,
            0.01,
            f"{n_out} non visible for being out of the domain",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================================================
# Main
# =========================================================
def predict(pic50_txt, model_dir, graph_dir, train_split_file, test_split_file, val_split_file, output_dir,
            log_callback, parameter_file):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_min, global_max = escala_global(pic50_txt)

    train_splits = load_split_txt(train_split_file)
    val_splits = load_split_txt(val_split_file)
    test_splits = load_split_txt(test_split_file)

    all_results = []
    all_labels_global = []
    all_preds_global = []
    parameter = parse_parameter_file(parameter_file)
    for split_idx in range(5):
        if log_callback: log_callback.info(f"\n===== SPLIT {split_idx:02d} =====")

        test_ids = test_splits[split_idx]
        test_set = URVGraphDataset(test_ids, graph_dir)
        test_loader = DataLoader(test_set, batch_size=parameter["batch_size"], shuffle=False)

        model = GIGN(parameter["node_dim"], parameter["hidden_dim"], parameter["drop_out"]).to(DEVICE)

        model_path = os.path.join(
            model_dir, f"split_{split_idx:02d}", "best_model.pt"
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        rmse, pearson, spearman, labels, preds = evaluate(model, test_loader)
        if log_callback:
            log_callback.info(f"RMSE: {rmse:.4f}")
            log_callback.info(f"Pearson: {pearson:.4f}")
            log_callback.info(f"Spearman: {spearman:.4f}")

        plot_split_scatter(
            labels, preds, f"Split_{split_idx:02d}", output_dir, rmse, pearson, global_min, global_max
        )

        all_results.append(
            {
                "Split": split_idx,
                "RMSE": rmse,
                "Pearson": pearson,
                "Spearman": spearman,
            }
        )

        all_labels_global.append(labels)
        all_preds_global.append(preds)

    # =========================================================
    # Summary
    # =========================================================

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(output_dir, "metrics_per_split.csv"), index=False
    )

    mean_row = results_df[["RMSE", "Pearson", "Spearman"]].mean()
    std_row = results_df[["RMSE", "Pearson", "Spearman"]].std()

    summary_df = pd.DataFrame(
        {
            "Metric": ["RMSE", "Pearson", "Spearman"],
            "Mean": mean_row.values,
            "Std": std_row.values,
        }
    )

    summary_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)

    if log_callback:
        log_callback.info("\n=== RESULTADOS FINALES ===")
        log_callback.info(summary_df)

    # =========================================================
    # Scatter global con promedios
    # =========================================================

    all_labels_global = np.concatenate(all_labels_global)
    all_preds_global = np.concatenate(all_preds_global)

    np.save(os.path.join(output_dir, "gign_labels.npy"), all_labels_global)
    np.save(os.path.join(output_dir, "gign_preds.npy"), all_preds_global)

    scatter_path = os.path.join(output_dir, "scatter_global.png")
    plot_global_scatter(
        all_labels_global,
        all_preds_global,
        scatter_path,
        mean_row["RMSE"],
        mean_row["Pearson"],
        global_min,
        global_max
    )
    if log_callback:
        log_callback.info(f"\nScatter global guardado en: {scatter_path}")
