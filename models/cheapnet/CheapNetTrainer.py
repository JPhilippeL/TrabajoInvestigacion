from time import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch_geometric.loader import DataLoader

from data_pipeline.common import load_split_txt, seed_everything
from data_pipeline.URVGraphDataset import URVGraphDataset
from job_config.cheapnet import CheapnetTrainerConfig
from models.cheapnet.CheapNet import CheapNet


class CheapNetTrainer:
    def __init__(self, config: CheapnetTrainerConfig):
        self.config = config

    def __num_clusters(self):
        q_lig = [0, 20, 28, 37, 177]
        q_pro = [0, 130, 156, 186, 500]
        q_i_lig = 2
        q_i_pro = 2
        num_clusters = [q_lig[q_i_lig], q_pro[q_i_pro]]
        return num_clusters

    @staticmethod
    def _val(model, dataloader, device, y_mean, y_std):
        model.eval()
        pred_list, label_list = [], []

        with torch.no_grad():
            for data in dataloader:
                data = data.to(device)
                pred = model(data)
                pred = pred * y_std + y_mean
                pred_list.append(pred.cpu().numpy())
                label_list.append(data.y.cpu().numpy())

        pred = np.concatenate(pred_list).reshape(-1)
        label = np.concatenate(label_list).reshape(-1)
        rmse = np.sqrt(mean_squared_error(label, pred))
        pearson = np.corrcoef(label, pred)[0, 1]
        model.train()
        return rmse, pearson

    def train(self, log_callback=None):
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        num_clusters = self.__num_clusters()
        train_splits = load_split_txt(self.config.train_split_file)
        val_splits = load_split_txt(self.config.val_split_file)
        test_splits = load_split_txt(self.config.test_split_file)

        start_training = time()

        split_best_val_rmses = []
        split_test_rmses = []
        split_test_pearsons = []

        for split_id in range(len(train_splits)):
            if log_callback:
                log_callback(f"Split {split_id}")

            seed_everything(split_id)

            train_ids = train_splits[split_id]
            val_ids = val_splits[split_id]
            test_ids = test_splits[split_id]

            train_set = URVGraphDataset(self.config.graphs_path, train_ids)
            val_set = URVGraphDataset(self.config.graphs_path, val_ids)
            test_set = URVGraphDataset(self.config.graphs_path, test_ids)

            train_loader = DataLoader(
                train_set,
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=True,
            )
            val_loader = DataLoader(val_set, batch_size=self.config.batch_size, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False)

            all_train_y = []

            for data in train_loader:
                all_train_y.append(data.y.view(-1))

            all_train_y = torch.cat(all_train_y).to(self.config.device)

            y_mean = all_train_y.mean()
            y_std = all_train_y.std()
            if log_callback:
                log_callback(f"Target mean:{y_mean.item()}")
                log_callback(f"Target std: {y_std.item()}")

                log_callback(f"Train samples: {len(train_set)}")
                log_callback(f"Test samples: {len(test_set)}")
                log_callback(f"Validation samples: {len(val_set)}")

            model = CheapNet(
                self.config.node_dim, self.config.hidden_dim, self.config.drop_out, num_clusters
            ).to(self.config.device)

            if log_callback:
                num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                log_callback(f"Trainable params: {num_params}")

            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
            )
            criterion = nn.MSELoss()

            best_rmse = float("inf")
            patience_counter = 0

            split_save_dir = self.config.output_path / f"split_{split_id:02d}"
            split_save_dir.mkdir(parents=True, exist_ok=True)

            best_model_path = split_save_dir / "best_model.pt"
            for epoch in range(self.config.epochs):
                model.train()
                epoch_loss = 0.0
                n_train = 0
                for data in train_loader:
                    data = data.to(self.config.device)

                    pred = model(data)
                    target = data.y.view(-1)
                    y = (target - y_mean) / y_std
                    loss = criterion(pred, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    current_batch_size = target.size(0)
                    epoch_loss += loss.item() * current_batch_size
                    n_train += current_batch_size

                train_rmse_norm = np.sqrt(epoch_loss / n_train)
                test_rmse, test_pr = self._val(
                    model, test_loader, self.config.device, y_mean, y_std
                )
                val_rmse, _ = self._val(model, val_loader, self.config.device, y_mean, y_std)
                train_rmse, train_pr = self._val(
                    model,
                    train_loader,
                    self.config.device,
                    y_mean,
                    y_std,
                )

                if log_callback:
                    log_callback(
                        f"Epoch {epoch:03d} | "
                        f"Train RMSE: {train_rmse:.4f} | "
                        f"Val RMSE: {val_rmse:.4f} | "
                        f"Test RMSE: {test_rmse:.4f} | "
                        f"Test Pearson: {test_pr:.4f}"
                    )
                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    patience_counter = 0

                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "split_id": split_id,
                            "best_epoch": epoch,
                            "best_val_rmse": best_rmse,
                            "y_mean": y_mean,
                            "y_std": y_std,
                            "config": {
                                "node_dim": self.config.node_dim,
                                "hidden_dim": self.config.hidden_dim,
                                "drop_out": self.config.drop_out,
                                "batch_size": self.config.batch_size,
                                "lr": self.config.lr,
                                "weight_decay": self.config.weight_decay,
                                "epochs": self.config.epochs,
                                "patience": self.config.patience,
                            },
                        },
                        best_model_path,
                    )
                    if log_callback:
                        log_callback(">>> Better model stored")

                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    if log_callback:
                        log_callback(">>> Early stopping activado")
                    break

            checkpoint = torch.load(
                best_model_path,
                map_location=self.config.device,
                weights_only=False,
            )

            model.load_state_dict(checkpoint["model_state_dict"])

            y_mean = checkpoint["y_mean"].to(self.config.device)
            y_std = checkpoint["y_std"].to(self.config.device)

            best_test_rmse, best_test_pr = self._val(
                model,
                test_loader,
                self.config.device,
                y_mean,
                y_std,
            )

            split_best_val_rmses.append(checkpoint["best_val_rmse"])
            split_test_rmses.append(best_test_rmse)
            split_test_pearsons.append(best_test_pr)

            if log_callback:
                log_callback(
                    f"Split {split_id:02d} | "
                    f"Best epoch: {checkpoint['best_epoch']} | "
                    f"Best Val RMSE: {checkpoint['best_val_rmse']:.4f} | "
                    f"Best Test RMSE: {best_test_rmse:.4f} | "
                    f"Best Test Pearson: {best_test_pr:.4f}"
                )

        end_train = time()
        train_time = end_train - start_training
        mean_val_rmse = float(np.mean(split_best_val_rmses))
        mean_test_rmse = float(np.mean(split_test_rmses))
        mean_test_pearson = float(np.mean(split_test_pearsons))

        std_val_rmse = float(np.std(split_best_val_rmses))
        std_test_rmse = float(np.std(split_test_rmses))
        std_test_pearson = float(np.std(split_test_pearsons))

        if log_callback:
            log_callback(f"\nIt took {train_time:.2f} seconds to train all splits.")
            log_callback("\nTraining completed for CheapNet.")
            log_callback(f"Mean Val RMSE: {mean_val_rmse:.4f}")
            log_callback(f"Mean Test RMSE: {mean_test_rmse:.4f}")
            log_callback(f"Mean Test Pearson: {mean_test_pearson:.4f}")

        return {
            "mean_val_rmse": mean_val_rmse,
            "std_val_rmse": std_val_rmse,
            "mean_test_rmse": mean_test_rmse,
            "std_test_rmse": std_test_rmse,
            "mean_test_pearson": mean_test_pearson,
            "std_test_pearson": std_test_pearson,
        }
