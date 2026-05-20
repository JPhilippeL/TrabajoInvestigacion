import matplotlib

from CheapNet_GUI.model.CheapNet_model import CheapNet

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from time import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from torch_geometric.loader import DataLoader

from CheapNet_GUI.model.utils import load_split_txt, URVGraphDataset, escala_global


def evaluate(model, dataloader, y_mean, y_std):
    model.eval()
    preds, labels = [], []
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    y_mean = y_mean.to(DEVICE)
    y_std = y_std.to(DEVICE)
    with torch.no_grad():
        for data in dataloader:
            data = data.to(DEVICE)
            pred = model(data).view(-1)
            pred = pred * y_std + y_mean
            target = data.y.view(-1)
            preds.append(pred.cpu().numpy())
            labels.append(target.cpu().numpy())

    preds = np.concatenate(preds).reshape(-1)
    labels = np.concatenate(labels).reshape(-1)

    rmse = np.sqrt(mean_squared_error(labels, preds))
    pearson = np.corrcoef(labels, preds)[0, 1]
    spearman = spearmanr(labels, preds)[0]

    return rmse, pearson, spearman, labels, preds


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
        f"CheapNet – {split_name}\nRMSE = {rmse:.3f} | Pearson = {pearson:.3f}"
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
        f"CheapNet - GLOBAL\nRMSE = {mean_rmse:.3f} | Pearson = {mean_pearson:.3f}"
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


def predict(pic50_txt, test_split_file, graph_dir, model_dir, output_dir, log_callback,
            batch_size=32):
    debut_prediction = time()
    all_results = []
    all_labels_global = []
    all_preds_global = []
    global_min, global_max = escala_global(pic50_txt)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    test_splits = load_split_txt(test_split_file)

    for split_idx in range(5):
        split_name = f"Split {split_idx:02d}"
        if log_callback:
            log_callback.info("\n==============================")
            log_callback.info(f"        {split_name}")
            log_callback.info("==============================")

        test_ids = test_splits[split_idx]

        test_set = URVGraphDataset(graph_dir, test_ids)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        split_model_dir = os.path.join(model_dir, f"split_{split_idx:02d}")
        model_path = os.path.join(split_model_dir, "best_model.pt")

        if not os.path.exists(model_path):
            if log_callback:
                log_callback.info(
                    f"[WARNING CHEAPNET] No directory with {model_path} in {split_model_dir}"
                )
            continue
        model_path = os.path.join(
            model_dir, f"split_{split_idx:02d}", "best_model.pt"
        )
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        config = checkpoint["config"]
        q_lig = [0, 20, 28, 37, 177]
        q_pro = [0, 130, 156, 186, 500]
        q_i_lig = 2
        q_i_pro = 2
        num_clusters = [q_lig[q_i_lig], q_pro[q_i_pro]]
        config = checkpoint["config"]
        if "node_dim" in config:
            node_dim = config["node_dim"]
        elif "NODE_DIM" in config:
            node_dim = config["NODE_DIM"]
        else:
            raise KeyError("node dimension not found")
        model = CheapNet(
            node_dim,
            config["hidden_dim"],
            config["drop_out"],
            num_clusters,
        ).to(DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])

        y_mean = checkpoint["y_mean"].to(DEVICE)
        y_std = checkpoint["y_std"].to(DEVICE)
        rmse, pearson, spearman, preds, labels = evaluate(model, test_loader, y_mean, y_std)
        if log_callback:
            log_callback.info(f"RMSE: {rmse:.4f}")
            log_callback.info(f"Pearson: {pearson:.4f}")
            log_callback.info(f"Spearman: {spearman:.4f}")

            log_callback.info(f"Pred mean: {preds.mean():.4f}")
            log_callback.info(f"Pred std: {preds.std():.4f}")
            log_callback.info(f"Label mean: {labels.mean():.4f}")
            log_callback.info(f"Label std: {labels.std():.4f}")
            log_callback.info(f"Pred min/max: {preds.min():.4f} / {preds.max():.4f}")
            log_callback.info(f"Label min/max: {labels.min():.4f} / {labels.max():.4f}")

        plot_split_scatter(labels, preds, split_name, output_dir, rmse, pearson, global_min, global_max)

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
    summary_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    if log_callback:
        log_callback.info("\n=== RESULTADOS FINALES ===")
        log_callback.info(summary_df)

    all_labels_global = np.concatenate(all_labels_global)
    all_preds_global = np.concatenate(all_preds_global)

    np.save(os.path.join(output_dir, "cheapnet_labels.npy"), all_labels_global)
    np.save(os.path.join(output_dir, "cheapnet_preds.npy"), all_preds_global)

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
    end_prediction = time()
    if log_callback:
        log_callback.info(f"it took {end_prediction - debut_prediction:.2f} seconds")
        log_callback.info(f"\nScatter global guardado en: {scatter_path}")
