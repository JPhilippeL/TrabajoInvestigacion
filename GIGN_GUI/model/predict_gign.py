import os
from time import time

import matplotlib
import numpy as np
import pandas as pd
import torch

from GIGN_GUI.model.gign_model import GIGN

# don't remove
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from torch_geometric.loader import DataLoader
from GIGN_GUI.model.utils import URVGraphDataset, load_split_txt, escala_global


def evaluate(model, dataloader):
    model.eval()
    preds, labels = [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            pred = model(data).view(-1)
            target = data.y.view(-1)
            preds.append(pred.cpu().numpy())
            labels.append(target.cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    rmse = np.sqrt(mean_squared_error(labels, preds))
    pearson = np.corrcoef(labels, preds)[0, 1]
    spearman = spearmanr(labels, preds)[0]

    return rmse, pearson, spearman, labels, preds


def plot_split_scatter(
        labels, preds, split_name, save_dir, rmse, pearson, global_min, global_max
):
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
        all_labels,
        all_preds,
        save_path,
        mean_rmse,
        mean_pearson,
        global_min,
        global_max,
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


def predict(
        pic50_txt,
        model_dir,
        graph_dir,
        test_split_file,
        output_dir,
        log_callback,
        batch_size=32

):
    debut_prediction = time()
    global_min, global_max = escala_global(pic50_txt)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_splits = load_split_txt(test_split_file)

    all_results = []
    all_labels_global = []
    all_preds_global = []
    for split_idx in range(len(test_splits)):
        if log_callback:
            log_callback.info(f"\n===== SPLIT {split_idx:02d} =====")

        test_ids = test_splits[split_idx]
        test_set = URVGraphDataset(graph_dir, test_ids)
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False
        )

        model_path = os.path.join(
            model_dir, f"split_{split_idx:02d}", "best_model.pt"
        )
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )

        config = checkpoint["config"]
        if "node_dim" in config:
            node_dim = config["node_dim"]
        elif "NODE_DIM" in config:
            node_dim = config["NODE_DIM"]
        else:
            raise KeyError("node dimension not found")

        model = GIGN(
            node_dim,
            config["hidden_dim"],
            config["drop_out"],
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        rmse, pearson, spearman, labels, preds = evaluate(model, test_loader)

        if log_callback:
            log_callback.info(f"RMSE: {rmse:.4f}")
            log_callback.info(f"Pearson: {pearson:.4f}")
            log_callback.info(f"Spearman: {spearman:.4f}")

        plot_split_scatter(
            labels,
            preds,
            f"Split_{split_idx:02d}",
            output_dir,
            rmse,
            pearson,
            global_min,
            global_max,
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

    summary_df.to_csv(
        os.path.join(output_dir, "metrics_summary.csv"), index=False
    )

    if log_callback:
        log_callback.info("\n=== RESULTADOS FINALES ===")
        log_callback.info(summary_df)

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
        global_max,
    )
    end_prediction = time()

    if log_callback:
        log_callback.info(f"prediction total time: {end_prediction - debut_prediction:.2f} seconds")
        log_callback.info(f"\nScatter global guardado en: {scatter_path}")
