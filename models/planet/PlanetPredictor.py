from time import time

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

from data_pipeline.planet_dataset import create_dataloaders_from_output
from job_config.planet.PlanetPredictionConfig import PlanetPredictionConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PlanetPredictor:
    def __init__(self, config: PlanetPredictionConfig):
        self.config = config

    def _log(self, log_callback, message):
        if log_callback is not None:
            log_callback(message)
        else:
            print(message)

    def _move_batch_to_device(self, batch):
        res_batch, mol_batch, targets = batch

        fresidues, res_map, res_scope, alpha_coordinates = res_batch
        fatoms, fbonds, agraph, bgraph, lig_scope = mol_batch

        device = self.config.device

        moved_res_batch = (
            fresidues.to(device),
            res_map.to(device) if isinstance(res_map, torch.Tensor) else res_map,
            res_scope,
            alpha_coordinates.to(device),
        )

        moved_mol_batch = (
            fatoms.to(device),
            fbonds.to(device),
            agraph.to(device),
            bgraph.to(device),
            lig_scope,
        )

        moved_targets = []
        for target in targets:
            if isinstance(target, torch.Tensor):
                moved_targets.append(target.to(device))
            else:
                moved_targets.append(target)

        return moved_res_batch, moved_mol_batch, tuple(moved_targets)

    def _safe_correlations(self, labels, preds):
        if len(labels) < 2:
            return 0.0, 0.0

        if np.std(labels) == 0 or np.std(preds) == 0:
            return 0.0, 0.0

        pearson = float(np.corrcoef(labels, preds)[0, 1])
        spearman = float(spearmanr(labels, preds)[0])

        if np.isnan(pearson):
            pearson = 0.0

        if np.isnan(spearman):
            spearman = 0.0

        return pearson, spearman

    def _evaluate(self, model, dataloader):
        model.eval()

        preds = []
        labels = []

        with torch.no_grad():
            for batch in dataloader:
                res_batch, mol_batch, targets = self._move_batch_to_device(batch)

                predictions = model(res_batch, mol_batch)

                predicted_affinities = predictions[2].detach().cpu().view(-1).numpy()
                pks = targets[2].detach().cpu().view(-1).numpy()
                pk_flags = targets[3].detach().cpu().view(-1).numpy()

                mask = pk_flags > 0

                if mask.sum() == 0:
                    continue

                preds.append(predicted_affinities[mask])
                labels.append(pks[mask])

        if not preds:
            raise RuntimeError("Aucune prédiction valide : tous les pk_flags sont à 0.")

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        rmse = float(np.sqrt(mean_squared_error(labels, preds)))
        pearson, spearman = self._safe_correlations(labels, preds)

        return rmse, pearson, spearman, labels, preds

    def plot_scatter(
        self,
        labels,
        preds,
        rmse,
        pearson,
        global_min,
        global_max,
    ):
        mask = (preds >= global_min) & (preds <= global_max)
        n_out = int((~mask).sum())

        plt.figure(figsize=(5, 5))
        plt.scatter(labels[mask], preds[mask], alpha=0.6)

        plt.xlim(global_min, global_max)
        plt.ylim(global_min, global_max)

        plt.xlabel("Real Value (pIC50)")
        plt.ylabel("Predicted Value (pIC50)")
        plt.title(
            f"PLANET {self.config.model_name} – core\n"
            f"RMSE = {rmse:.3f} | Pearson = {pearson:.3f}"
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

        save_path = self.config.output_path / "scatter_core.png"
        plt.savefig(save_path)
        plt.close()

        return save_path

    def _load_model(self):
        model_path = self.config.model_path / "best_model.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")

        checkpoint = torch.load(
            model_path,
            map_location=self.config.device,
            weights_only=False,
        )

        checkpoint_config = checkpoint["config"]

        model = self.config.model(
            feature_dims=checkpoint_config["feature_dims"],
            nheads=checkpoint_config["nheads"],
            key_dims=checkpoint_config["key_dims"],
            value_dims=checkpoint_config["value_dims"],
            pro_update_inters=checkpoint_config["pro_update_inters"],
            lig_update_iters=checkpoint_config["lig_update_iters"],
            pro_lig_update_iters=checkpoint_config["pro_lig_update_iters"],
            device=self.config.device,
        ).to(self.config.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model_name = checkpoint.get("model_name", self.config.model_name)

        return model, model_name, model_path

    def predict(self, log_callback=None):
        start_prediction = time()
        self.config.output_path.mkdir(parents=True, exist_ok=True)

        self._log(log_callback, "=== PLANET prediction on core.pkl ===")

        core_pkl = self.config.data_output_path / "metadata" / "pkl" / "core.pkl"

        if not core_pkl.exists():
            raise FileNotFoundError(f"core.pkl introuvable : {core_pkl}")

        _, _, core_loader = create_dataloaders_from_output(
            output_dir=self.config.data_output_path,
            planet_root=None,
            batch_size=self.config.batch_size,
            seed=self.config.seed,
            decoy_flag=False,
            num_workers=self.config.num_workers,
        )

        model, model_name, model_path = self._load_model()

        self._log(log_callback, f"Model loaded from: {model_path}")

        rmse, pearson, spearman, labels, preds = self._evaluate(
            model=model,
            dataloader=core_loader,
        )

        self._log(log_callback, f"RMSE: {rmse:.4f}")
        self._log(log_callback, f"Pearson: {pearson:.4f}")
        self._log(log_callback, f"Spearman: {spearman:.4f}")

        global_min = float(min(labels.min(), preds.min()) - 0.2)
        global_max = float(max(labels.max(), preds.max()) + 0.2)

        axis_path = self.config.output_path / "global_axis.npy"
        np.save(axis_path, [global_min, global_max])

        scatter_path = self.plot_scatter(
            labels=labels,
            preds=preds,
            rmse=rmse,
            pearson=pearson,
            global_min=global_min,
            global_max=global_max,
        )

        metrics_df = pd.DataFrame(
            [
                {
                    "Split": "core",
                    "RMSE": rmse,
                    "Pearson": pearson,
                    "Spearman": spearman,
                }
            ]
        )

        metrics_csv = self.config.output_path / f"metrics_core_{model_name}.csv"
        metrics_df.to_csv(metrics_csv, index=False)

        summary_df = pd.DataFrame(
            {
                "Metric": ["RMSE", "Pearson", "Spearman"],
                "Mean": [rmse, pearson, spearman],
                "Std": [0.0, 0.0, 0.0],
            }
        )

        summary_csv = self.config.output_path / f"metrics_summary_{model_name}.csv"
        summary_df.to_csv(summary_csv, index=False)

        labels_path = self.config.output_path / f"planet_{model_name}_labels.npy"
        preds_path = self.config.output_path / f"planet_{model_name}_preds.npy"

        np.save(labels_path, labels)
        np.save(preds_path, preds)

        end_prediction = time()
        prediction_time = float(end_prediction - start_prediction)

        self._log(log_callback, "\n=== FINAL RESULT ===")
        self._log(log_callback, str(summary_df))
        self._log(log_callback, f"Prediction total time: {prediction_time:.2f} seconds")
        self._log(log_callback, f"Metrics CSV: {metrics_csv}")
        self._log(log_callback, f"Summary CSV: {summary_csv}")
        self._log(log_callback, f"Scatter: {scatter_path}")

        return {
            "rmse": rmse,
            "pearson": pearson,
            "spearman": spearman,
            "prediction_time": prediction_time,
            "metrics_csv": str(metrics_csv),
            "summary_csv": str(summary_csv),
            "labels_path": str(labels_path),
            "preds_path": str(preds_path),
            "axis_path": str(axis_path),
            "scatter_path": str(scatter_path),
            "model_path": str(model_path),
        }