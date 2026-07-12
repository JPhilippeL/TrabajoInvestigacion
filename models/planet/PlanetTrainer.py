from pathlib import Path
from time import time

import numpy as np
import torch

from data_pipeline.common import seed_everything
from data_pipeline.planet_dataset import create_dataloaders_from_output
from job_config.planet.PlanetTrainerConfig import PlanetTrainerConfig


class PlanetTrainer:
    def __init__(self, config: PlanetTrainerConfig):
        self.config = config
        self.global_step = 0

    def _log(self, log_callback, message):
        if log_callback is not None:
            log_callback(message)
        else:
            print(message)

    def _safe_pearson(self, pred, label):
        pred = np.asarray(pred).reshape(-1)
        label = np.asarray(label).reshape(-1)

        if len(pred) < 2:
            return 0.0

        if np.std(pred) == 0 or np.std(label) == 0:
            return 0.0

        return float(np.corrcoef(pred, label)[0, 1])

    def _move_batch_to_device(self, batch):
        res_batch, mol_batch, targets = batch

        fresidues, res_map, res_scope, alpha_coordinates = res_batch
        fatoms, fbonds, agraph, bgraph, lig_scope = mol_batch

        device = self.config.device

        res_batch = (
            fresidues.to(device),
            res_map.to(device) if isinstance(res_map, torch.Tensor) else res_map,
            res_scope,
            alpha_coordinates.to(device),
        )

        mol_batch = (
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

        return res_batch, mol_batch, tuple(moved_targets)

    def _extract_affinity(self, predictions, targets):
        predicted_affinities = predictions[2].detach().cpu().view(-1).numpy()
        pks = targets[2].detach().cpu().view(-1).numpy()
        pk_flags = targets[3].detach().cpu().view(-1).numpy()

        mask = pk_flags > 0

        return predicted_affinities[mask], pks[mask]

    def _eval(self, model, dataloader):
        model.eval()

        lig_losses = []
        pro_lig_losses = []
        affinity_losses = []

        lig_accs = []
        pro_lig_accs = []
        affinity_maes = []

        pred_list = []
        label_list = []

        skipped_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                try:
                    res_batch, mol_batch, targets = self._move_batch_to_device(batch)

                    predictions = model(res_batch, mol_batch)

                    lig_loss, pro_lig_loss, affinity_loss = model.compute_loss(
                        predictions,
                        targets,
                        res_batch,
                        mol_batch,
                    )

                    lig_acc, pro_lig_acc, affinity_mae = model.compute_metrics(
                        predictions,
                        targets,
                    )

                    pred, label = self._extract_affinity(predictions, targets)

                    if len(label) > 0:
                        pred_list.append(pred)
                        label_list.append(label)

                    lig_losses.append(float(lig_loss.item()))
                    pro_lig_losses.append(float(pro_lig_loss.item()))
                    affinity_losses.append(float(affinity_loss.item()))

                    lig_accs.append(float(lig_acc.item()))
                    pro_lig_accs.append(float(pro_lig_acc.item()))
                    affinity_maes.append(float(affinity_mae.item()))

                except Exception as exc:
                    skipped_batches += 1
                    print(f"Skipping eval batch: {type(exc).__name__}: {exc}")

        if pred_list:
            pred = np.concatenate(pred_list)
            label = np.concatenate(label_list)

            rmse = float(np.sqrt(np.mean((pred - label) ** 2)))
            pearson = self._safe_pearson(pred, label)
        else:
            rmse = float("inf")
            pearson = 0.0

        model.train()

        return {
            "rmse": rmse,
            "pearson": pearson,
            "lig_interaction_loss": float(np.mean(lig_losses)) if lig_losses else float("nan"),
            "pro_lig_interaction_loss": float(np.mean(pro_lig_losses))
            if pro_lig_losses
            else float("nan"),
            "affinity_loss": float(np.mean(affinity_losses)) if affinity_losses else float("nan"),
            "lig_interaction_acc": float(np.mean(lig_accs)) if lig_accs else float("nan"),
            "pro_lig_interaction_acc": float(np.mean(pro_lig_accs))
            if pro_lig_accs
            else float("nan"),
            "affinity_mae": float(np.mean(affinity_maes)) if affinity_maes else float("nan"),
            "skipped_batches": skipped_batches,
        }

    def _initialize_weights(self, model):
        for param in model.parameters():
            if param.dim() == 1:
                torch.nn.init.constant_(param, 0)
            else:
                torch.nn.init.xavier_uniform_(param)

    def _build_model(self):
        model = self.config.model(
            feature_dims=self.config.feature_dims,
            nheads=self.config.nheads,
            key_dims=self.config.key_dims,
            value_dims=self.config.value_dims,
            pro_update_inters=self.config.pro_update_inters,
            lig_update_iters=self.config.lig_update_iters,
            pro_lig_update_iters=self.config.pro_lig_update_iters,
            device=self.config.device,
        )

        self._initialize_weights(model)

        return model.to(self.config.device)

    def _compute_total_loss(self, lig_loss, pro_lig_loss, affinity_loss):
        beta = 0.0 if self.global_step <= self.config.beta_start_step else 1.0
        total_loss = lig_loss + pro_lig_loss + beta * affinity_loss
        return total_loss, beta

    def train(self, log_callback=None):
        self.config.output_path.mkdir(parents=True, exist_ok=True)

        seed_everything(self.config.seed)

        train_loader, val_loader, test_loader = create_dataloaders_from_output(
            output_dir=self.config.data_output_path,
            planet_root=None,
            batch_size=self.config.batch_size,
            seed=self.config.seed,
            decoy_flag=True,
            num_workers=self.config.num_workers,
        )

        split_save_dir = self.config.output_path
        split_save_dir.mkdir(parents=True, exist_ok=True)

        best_model_path = split_save_dir / "best_model.pt"

        model = self._build_model()

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self._log(log_callback, f"Trainable params: {trainable_params}")

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        best_val_rmse = float("inf")
        best_epoch = -1
        patience_counter = 0

        starting_time = time()

        for epoch in range(self.config.epochs):
            model.train()

            epoch_losses = []
            epoch_affinity_preds = []
            epoch_affinity_labels = []

            skipped_train_batches = 0

            for batch in train_loader:
                try:
                    res_batch, mol_batch, targets = self._move_batch_to_device(batch)
                    optimizer.zero_grad()

                    predictions = model(res_batch, mol_batch)

                    lig_loss, pro_lig_loss, affinity_loss = model.compute_loss(
                        predictions,
                        targets,
                        res_batch,
                        mol_batch,
                    )

                    total_loss, beta = self._compute_total_loss(
                        lig_loss,
                        pro_lig_loss,
                        affinity_loss,
                    )

                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        self.config.clip_norm,
                    )

                    optimizer.step()

                    pred, label = self._extract_affinity(predictions, targets)

                    if len(label) > 0:
                        epoch_affinity_preds.append(pred)
                        epoch_affinity_labels.append(label)

                    epoch_losses.append(float(total_loss.item()))
                    self.global_step += 1

                except Exception as exc:
                    skipped_train_batches += 1
                    self._log(
                        log_callback,
                        f"Skipping train batch: {type(exc).__name__}: {exc}",
                    )

            if epoch_affinity_preds:
                train_pred = np.concatenate(epoch_affinity_preds)
                train_label = np.concatenate(epoch_affinity_labels)
                train_rmse = float(np.sqrt(np.mean((train_pred - train_label) ** 2)))
            else:
                train_rmse = float("inf")

            val_metrics = self._eval(model, val_loader)
            val_rmse = val_metrics["rmse"]
            val_pearson = val_metrics["pearson"]

            self._log(
                log_callback,
                f"Epoch {epoch:03d} | "
                f"Train RMSE: {train_rmse:.4f} | "
                f"Val RMSE: {val_rmse:.4f} | "
                f"Val Pearson: {val_pearson:.4f} | "
                f"Skipped train: {skipped_train_batches} | "
                f"Skipped val: {val_metrics['skipped_batches']}",
            )

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                patience_counter = 0

                torch.save(
                    {
                        "model_name": self.config.model_name,
                        "model_state_dict": model.state_dict(),
                        "best_epoch": best_epoch,
                        "best_val_rmse": best_val_rmse,
                        "config": {
                            "feature_dims": self.config.feature_dims,
                            "nheads": self.config.nheads,
                            "key_dims": self.config.key_dims,
                            "value_dims": self.config.value_dims,
                            "pro_update_inters": self.config.pro_update_inters,
                            "lig_update_iters": self.config.lig_update_iters,
                            "pro_lig_update_iters": self.config.pro_lig_update_iters,
                            "batch_size": self.config.batch_size,
                            "lr": self.config.lr,
                            "weight_decay": self.config.weight_decay,
                            "patience": self.config.patience,
                            "epochs": self.config.epochs,
                            "clip_norm": self.config.clip_norm,
                            "beta_start_step": self.config.beta_start_step,
                        },
                    },
                    best_model_path,
                )

                self._log(log_callback, ">>> Better model stocked")

            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                self._log(log_callback, ">>> Early Stopped")
                break
        end_training = time()
        if not best_model_path.exists():
            raise RuntimeError(
                "No valid best model was saved. "
                "Check validation loader, pk_flags, skipped batches, or tensorization errors."
            )
        checkpoint = torch.load(
            best_model_path,
            map_location=self.config.device,
            weights_only=False,
        )

        config = checkpoint["config"]

        best_model = self.config.model(
            feature_dims=config["feature_dims"],
            nheads=config["nheads"],
            key_dims=config["key_dims"],
            value_dims=config["value_dims"],
            pro_update_inters=config["pro_update_inters"],
            lig_update_iters=config["lig_update_iters"],
            pro_lig_update_iters=config["pro_lig_update_iters"],
            device=self.config.device,
        ).to(self.config.device)

        best_model.load_state_dict(checkpoint["model_state_dict"])

        train_metrics = self._eval(best_model, train_loader)
        val_metrics = self._eval(best_model, val_loader)
        test_metrics = self._eval(best_model, test_loader)

        train_rmse = train_metrics["rmse"]
        val_rmse = val_metrics["rmse"]
        test_rmse = test_metrics["rmse"]
        test_pearson = test_metrics["pearson"]

        self._log(
            log_callback,
            f"[PLANET] "
            f"Best epoch: {best_epoch} | "
            f"Best Train RMSE: {train_rmse:.4f} | "
            f"Best Val RMSE: {val_rmse:.4f} | "
            f"Test RMSE: {test_rmse:.4f} | "
            f"Test Pearson: {test_pearson:.4f}",
        )

        self._log(log_callback, "Training finished")
        self._log(log_callback, f"Train total time: {end_training - starting_time:.2f} seconds")

        return {
            "train_rmse": float(train_rmse),
            "val_rmse": float(val_rmse),
            "test_rmse": float(test_rmse),
            "test_pearson": float(test_pearson),
            "best_epoch": int(best_epoch),
            "best_val_rmse": float(best_val_rmse),
            "training_time": float(end_training - starting_time),
            "best_model_path": str(best_model_path),
        }
