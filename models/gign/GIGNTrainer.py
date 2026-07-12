import os
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch_geometric.loader import DataLoader

from data_pipeline.common import load_split_txt, seed_everything
from data_pipeline.URVGraphDataset import URVGraphDataset
from job_config.gign.GIGNTrainerConfig import GIGNTrainerConfig
from models.gign.GIGN import GIGN


class GIGNTrainer:
    def __init__(self, config: GIGNTrainerConfig):
        self.config = config

    def _val(self, model, dataloader):
        model.eval()
        pred_list, label_list = [], []

        for data in dataloader:
            data = data.to(self.config.device)
            with torch.no_grad():
                pred = model(data)

            pred_list.append(pred.cpu().numpy())
            label_list.append(data.y.cpu().numpy())

        pred = np.concatenate(pred_list)
        label = np.concatenate(label_list)

        rmse = np.sqrt(mean_squared_error(label, pred))
        pearson = np.corrcoef(pred, label)[0, 1]

        model.train()
        return rmse, pearson

    def train(self, log_callback=None):
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        debut_training = time()
        train_splits = load_split_txt(self.config.train_split_file)
        val_splits = load_split_txt(self.config.val_split_file)
        test_splits = load_split_txt(self.config.test_split_file)

        split_best_val_rmses = []
        split_test_rmses = []
        split_test_pearsons = []
        split_train_rmses = []

        for split_id in range(len(train_splits)):
            if log_callback:
                log_callback(f"SPLIT {split_id}")

            seed_everything(split_id)

            train_ids = train_splits[split_id]
            val_ids = val_splits[split_id]
            test_ids = test_splits[split_id]

            train_set = URVGraphDataset(self.config.graphs_path, train_ids)
            val_set = URVGraphDataset(self.config.graphs_path, val_ids)
            test_set = URVGraphDataset(self.config.graphs_path, test_ids)

            train_loader = DataLoader(
                train_set, batch_size=self.config.batch_size, shuffle=True, drop_last=False
            )
            val_loader = DataLoader(val_set, batch_size=self.config.batch_size, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False)
            if log_callback:
                log_callback(f"Train samples: {len(train_set)}")
                log_callback(f"Test samples: {len(test_set)}")
                log_callback(f"Validation samples: {len(val_set)}")

            model = GIGN(self.config.node_dim, self.config.hidden_dim, self.config.drop_out).to(
                self.config.device
            )
            if log_callback:
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                log_callback(f"Trainable params: {trainable_params}")

            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
            )

            criterion = nn.MSELoss()

            best_rmse = float("inf")
            best_epoch = 0
            patience_counter = 0

            split_save_dir = Path(self.config.output_path) / f"split_{split_id:02d}"
            os.makedirs(split_save_dir, exist_ok=True)
            best_model_path = os.path.join(split_save_dir, "best_model.pt")

            for epoch in range(self.config.epochs):
                model.train()
                epoch_loss = 0.0
                n_train = 0
                for data in train_loader:
                    data = data.to(self.config.device)

                    pred = model(data)
                    target = data.y.view(-1)
                    loss = criterion(pred, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_size_current = target.size(0)
                    epoch_loss += loss.item() * batch_size_current
                    n_train += batch_size_current

                train_rmse = np.sqrt(epoch_loss / n_train)
                test_rmse, test_pr = self._val(model, test_loader)
                val_rmse, val_pr = self._val(model, val_loader)

                if log_callback:
                    log_callback(
                        f"Epoch {epoch:03d} | "
                        f"Train RMSE: {train_rmse:.4f} | "
                        f"Test RMSE: {test_rmse:.4f} | "
                        f"Val RMSE: {val_rmse:.4f} | "
                        f"Pearson: {test_pr:.4f}"
                    )

                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_epoch = epoch
                    patience_counter = 0

                    torch.save(
                        {
                            "split_id": split_id,
                            "best_epoch": best_epoch,
                            "best_val_rmse": best_rmse,
                            "model_state_dict": model.state_dict(),
                            "config": {
                                "epochs": epoch,
                                "node_dim": self.config.node_dim,
                                "hidden_dim": self.config.hidden_dim,
                                "batch_size": self.config.batch_size,
                                "lr": self.config.lr,
                                "weight_decay": self.config.weight_decay,
                                "patience": self.config.patience,
                                "drop_out": self.config.drop_out,
                            },
                        },
                        best_model_path,
                    )
                    if log_callback:
                        log_callback(">>> Better model stocked")

                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    if log_callback:
                        log_callback(">>> Early stopping activado")
                    break

            checkpoint = torch.load(
                best_model_path, map_location=self.config.device, weights_only=False
            )

            model.load_state_dict(checkpoint["model_state_dict"])

            best_train_rmse, best_train_pr = self._val(model, train_loader)
            best_test_rmse, best_test_pr = self._val(model, test_loader)

            split_best_val_rmses.append(checkpoint["best_val_rmse"])
            split_train_rmses.append(best_train_rmse)
            split_test_rmses.append(best_test_rmse)
            split_test_pearsons.append(best_test_pr)

            if log_callback:
                log_callback(
                    f"Split {split_id:02d} | "
                    f"Best epoch: {checkpoint['best_epoch']} | "
                    f"Best Train RMSE: {best_train_rmse:.4f} | "
                    f"Best Val RMSE: {checkpoint['best_val_rmse']:.4f} | "
                    f"Best Test RMSE: {best_test_rmse:.4f} | "
                    f"Best Test Pearson: {best_test_pr:.4f}"
                )
        if log_callback:
            log_callback("\nTraining completed for all splits.")
        mean_train_rmse = float(np.mean(split_train_rmses))
        mean_val_rmse = float(np.mean(split_best_val_rmses))
        mean_test_rmse = float(np.mean(split_test_rmses))
        mean_test_pearson = float(np.mean(split_test_pearsons))

        std_train_rmse = float(np.std(split_train_rmses))
        std_val_rmse = float(np.std(split_best_val_rmses))
        std_test_rmse = float(np.std(split_test_rmses))
        std_test_pearson = float(np.std(split_test_pearsons))
        end_training = time()
        if log_callback:
            log_callback("\nTraining completed for all splits.")
            log_callback(f"Train total time: {end_training - debut_training:.2f} seconds")

            log_callback(f"Mean Train RMSE: {mean_train_rmse:.4f}")
            log_callback(f"Std Train RMSE: {std_train_rmse:.4f}")

            log_callback(f"Mean Val RMSE: {mean_val_rmse:.4f}")
            log_callback(f"Std Val RMSE: {std_val_rmse:.4f}")

            log_callback(f"Mean Test RMSE: {mean_test_rmse:.4f}")
            log_callback(f"Std Test RMSE: {std_test_rmse:.4f}")

            log_callback(f"Mean Test Pearson: {mean_test_pearson:.4f}")
            log_callback(f"Std Test Pearson: {std_test_pearson:.4f}")
        return {
            "mean_train_rmse": mean_train_rmse,
            "std_train_rmse": std_train_rmse,
            "mean_val_rmse": mean_val_rmse,
            "std_val_rmse": std_val_rmse,
            "mean_test_rmse": mean_test_rmse,
            "std_test_rmse": std_test_rmse,
            "mean_test_pearson": mean_test_pearson,
            "std_test_pearson": std_test_pearson,
        }
