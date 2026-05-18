import os
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from GIGN_GUI.model.gign_model import GIGN
from GIGN_GUI.model.utils import (
    URVGraphDataset,
    load_split_txt,
    seed_everything,
    val,
    write_hyperparameter_into_a_file,
)


def train_gign(
        train_file,
        test_file,
        val_file,
        epochs,
        seed,
        node_dim,
        hidden_dim,
        batch_size,
        lr,
        weight_decay,
        patience,
        drop_out,
        save_dir,
        log_callback,
        graph_dir,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)
    debut_training = time()
    train_splits = load_split_txt(train_file)
    val_splits = load_split_txt(val_file)
    test_splits = load_split_txt(test_file)

    split_best_val_rmses = []
    split_test_rmses = []
    split_test_pearsons = []
    for split_id in range(len(train_splits)):
        if log_callback:
            log_callback.info(f"SPLIT {split_id}")

        seed_everything(split_id + seed)

        train_ids = train_splits[split_id]
        val_ids = val_splits[split_id]
        test_ids = test_splits[split_id]

        train_set = URVGraphDataset(graph_dir, train_ids)
        val_set = URVGraphDataset(graph_dir, val_ids)
        test_set = URVGraphDataset(graph_dir, test_ids)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        if log_callback:
            log_callback.info(f"Train samples: {len(train_set)}")
            log_callback.info(f"Test samples: {len(test_set)}")
            log_callback.info(f"Validation samples: {len(val_set)}")

        model = GIGN(node_dim, hidden_dim, drop_out).to(device)
        if log_callback:
            log_callback.info(
                f"Trainable params:",
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}",
            )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        criterion = nn.MSELoss()

        best_rmse = float("inf")
        best_epoch = 0
        patience_counter = 0

        split_save_dir = os.path.join(save_dir, f"split_{split_id:02d}")
        os.makedirs(split_save_dir, exist_ok=True)
        best_model_path = os.path.join(split_save_dir, "best_model.pt")

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_train = 0
            for data in train_loader:
                data = data.to(device)

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
            test_rmse, test_pr = val(model, test_loader, device)
            val_rmse, val_pr = val(model, val_loader, device)

            if log_callback:
                log_callback.info(
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
                            "node_dim": node_dim,
                            "hidden_dim": hidden_dim,
                            "batch_size": batch_size,
                            "lr": lr,
                            "weight_decay": weight_decay,
                            "patience": patience,
                            "drop_out": drop_out,
                        },
                    },
                    best_model_path,
                )
                if log_callback:
                    log_callback.info(">>> Better model stocked")

            else:
                patience_counter += 1

            if patience_counter >= patience:
                if log_callback:
                    log_callback.info(">>> Early stopping activado")
                break

        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_test_rmse, best_test_pr = val(model, test_loader, device)
        split_best_val_rmses.append(checkpoint["best_val_rmse"])
        split_test_rmses.append(best_test_rmse)
        split_test_pearsons.append(best_test_pr)

        if log_callback:
            log_callback.info(
                f"Split {split_id:02d} | "
                f"Best epoch: {checkpoint['best_epoch']} | "
                f"Best Val RMSE: {checkpoint['best_val_rmse']:.4f} | "
                f"Best Test RMSE: {best_test_rmse:.4f} | "
                f"Best Test Pearson: {best_test_pr:.4f}"
            )
    if log_callback:
        log_callback.info("\nTraining completed for all splits.")

    mean_val_rmse = float(np.mean(split_best_val_rmses))
    mean_test_rmse = float(np.mean(split_test_rmses))
    mean_test_pearson = float(np.mean(split_test_pearsons))
    end_training = time()
    if log_callback:
        log_callback.info("\nTraining completed for all splits.")
        log_callback.info(f"Train total time: {end_training - debut_training:.2f} seconds")
        log_callback.info(f"Mean Val RMSE: {mean_val_rmse:.4f}")
        log_callback.info(f"Mean Test RMSE: {mean_test_rmse:.4f}")
        log_callback.info(f"Mean Test Pearson: {mean_test_pearson:.4f}")
        log_callback.info(
            f"The parameters used are stored in {save_dir}/hyperparameters_gign.txt"
        )
    write_hyperparameter_into_a_file(
        epochs,
        node_dim,
        hidden_dim,
        drop_out,
        batch_size,
        lr,
        weight_decay,
        patience,
        os.path.join(save_dir, "hyperparameters_gign.txt"),
    )
    return {
        "mean_val_rmse": mean_val_rmse,
        "mean_test_rmse": mean_test_rmse,
        "mean_test_pearson": mean_test_pearson,
    }
