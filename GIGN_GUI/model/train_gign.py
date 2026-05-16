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

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        if log_callback:
            log_callback.info(f"Train samples: {len(train_set)}")
            log_callback.info(f"Test samples: {len(test_set)}")
            log_callback.info(f"Validation samples: {len(val_set)}")

        model = GIGN(node_dim, hidden_dim, drop_out).to(device)
        if log_callback:
            log_callback.info(
                "Trainable params:",
                sum(p.numel() for p in model.parameters() if p.requires_grad),
            )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        criterion = nn.MSELoss()

        best_rmse = float("inf")
        patience = patience
        patience_counter = 0

        split_save_dir = os.path.join(save_dir, f"split_{split_id:02d}")
        os.makedirs(split_save_dir, exist_ok=True)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            for data in train_loader:
                data = data.to(device)

                pred = model(data)
                loss = criterion(pred, data.y.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * data.y.size(0)

            train_rmse = np.sqrt(epoch_loss / len(train_set))
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
                patience_counter = 0

                torch.save(
                    model,
                    os.path.join(split_save_dir, "best_model.pt"),
                )
                if log_callback:
                    log_callback.info(">>> Better model stocked")

            else:
                patience_counter += 1

            if patience_counter >= patience:
                if log_callback:
                    log_callback.info(">>> Early stopping activado")
                break

        if log_callback:
            log_callback.info(
                f"Best RMSE split {split_id:02d}: {best_rmse:.4f}"
            )

    if log_callback:
        log_callback.info("\nTraining completed for all splits.")

    end_training = time()
    if log_callback:
        log_callback.info(f"train total time: {end_training - debut_training:.2f} seconds")
        log_callback.info(f"The parameters used are stored in {save_dir}/hyperparameters_gign.txt")
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
