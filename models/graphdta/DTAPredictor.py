from time import time

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from torch_geometric.loader import DataLoader

from data_pipeline.common import load_split_txt
from data_pipeline.URVGraphDataset import URVGraphDataset
from job_config.graphdta.DTAPredictionConfig import DTAPredictionConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class DTAPredictor:
    def __init__(self, config: DTAPredictionConfig):
        self.config = config

    def _evaluate(self, model, dataloader):
        model.eval()
        preds = []
        labels = []

        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.config.device)
                pred = model(data).view(-1)
                target = data.y.view(-1).float()
                preds.append(pred.cpu().numpy())
                labels.append(target.cpu().numpy())

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        rmse = np.sqrt(mean_squared_error(labels, preds))
        pearson = np.corrcoef(labels, preds)[0, 1]
        spearman = spearmanr(labels, preds)[0]
        return rmse, pearson, spearman, labels, preds

    def plot_split_scatter(self, labels, preds, split_name, rmse, pearson, global_min, global_max):
        mask = (preds >= global_min) & (preds <= global_max)
        n_out = (~mask).sum()

        plt.figure(figsize=(5, 5))
        plt.scatter(labels[mask], preds[mask], alpha=0.6)

        plt.xlim(global_min, global_max)
        plt.ylim(global_min, global_max)

        plt.xlabel("Real Value (pIC50)")
        plt.ylabel("Predicted Value (pIC50)")
        plt.title(
            f"GraphDTA {self.config.model_name} – {split_name}\nRMSE = {rmse:.3f} | Pearson = {pearson:.3f}"
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
        save_path = self.config.output_path / f"scatter_{split_name}.png"
        plt.savefig(save_path)
        plt.close()

    def plot_global_scatter(
        self,
        all_labels,
        all_preds,
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

        plt.xlabel("Real Value (pIC50)")
        plt.ylabel("Predicted Value (pIC50)")
        plt.title(
            f"GraphDTA {self.config.model_name} - GLOBAL\nRMSE = {mean_rmse:.3f} | Pearson = {mean_pearson:.3f}"
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
        save_path = self.config.output_path / "scatter_global.png"
        plt.savefig(save_path)
        plt.close()

    def escala_global(self):
        labels = np.loadtxt(self.config.pic50_path, usecols=1)

        real_min = labels.min()
        real_max = labels.max()

        margin = 0.2
        global_min = real_min - margin
        global_max = real_max + margin

        print("Global axis limits:", global_min, global_max)

        np.save(self.config.output_path / "global_axis.npy", [global_min, global_max])

        return global_min, global_max

    def predict(self, log_callback=None):
        debut_prediction = time()
        global_min, global_max = self.escala_global()
        self.config.output_path.mkdir(parents=True, exist_ok=True)

        test_splits = load_split_txt(self.config.test_split_file)

        all_results = []
        all_labels_global = []
        all_preds_global = []
        for split_idx in range(len(test_splits)):
            if log_callback:
                log_callback(f"\n===== SPLIT {split_idx:02d} =====")

            test_ids = test_splits[split_idx]
            test_set = URVGraphDataset(self.config.graphs_path, test_ids)
            test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False)

            model_path = self.config.model_path / f"split_{split_idx:02d}" / "best_model.pt"
            checkpoint = torch.load(
                model_path,
                map_location=self.config.device,
                weights_only=False,
            )

            config = checkpoint["config"]
            model_name = checkpoint["model_name"]
            n_filter = config["n_filters"]
            dropout = config["dropout"]
            model = self.config.model(
                n_filters=n_filter,
                dropout=dropout,
            ).to(self.config.device)

            model.load_state_dict(checkpoint["model_state_dict"])

            rmse, pearson, spearman, labels, preds = self._evaluate(model, test_loader)
            if log_callback:
                log_callback(f"RMSE: {rmse:.4f}")
                log_callback(f"Pearson: {pearson:.4f}")
                log_callback(f"Spearman: {spearman:.4f}")

            self.plot_split_scatter(
                labels,
                preds,
                f"Split_{split_idx:02d}",
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
        out_csv = self.config.output_path / f"metrics_per_split_{model_name}.csv"
        results_df.to_csv(str(out_csv), index=False)

        mean_row = results_df[["RMSE", "Pearson", "Spearman"]].mean()
        std_row = results_df[["RMSE", "Pearson", "Spearman"]].std()

        summary_df = pd.DataFrame(
            {
                "Metric": ["RMSE", "Pearson", "Spearman"],
                "Mean": mean_row.values,
                "Std": std_row.values,
            }
        )
        csv_global = self.config.output_path / f"metrics_summary_{model_name}.csv"
        summary_df.to_csv(str(csv_global), index=False)

        if log_callback:
            log_callback("\n=== FINAL RESULT ===")
            log_callback(summary_df)

        all_labels_global = np.concatenate(all_labels_global)
        all_preds_global = np.concatenate(all_preds_global)

        np.save(self.config.output_path / f"dta_{model_name}_labels.npy", all_labels_global)
        np.save(self.config.output_path / f"dta_{model_name}_preds.npy", all_preds_global)

        scatter_path = self.config.output_path / "scatter_global.png"
        self.plot_global_scatter(
            all_labels_global,
            all_preds_global,
            mean_row["RMSE"],
            mean_row["Pearson"],
            global_min,
            global_max,
            model_name,
        )
        end_prediction = time()

        if log_callback:
            log_callback(f"prediction total time: {end_prediction - debut_prediction:.2f} seconds")
            log_callback(f"\nScatter global in: {scatter_path}")
