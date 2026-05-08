import os
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from CheapNet_GUI.model.CheapNet_model import CheapNet
from CheapNet_GUI.model.utils import load_split_txt, seed_everything, val, URVGraphDataset, \
    write_hyperparameter_in_a_file


def train_cheapnet(save_dir, train_split_file, val_split_file, test_split_file, graph_dir, batch_size, lr, hidden_dim,
                   node_dim, weight_decay, epochs, drop_rate, patience, log_callback):
    # Quantiles of train set
    q_lig = [0, 20, 28, 37, 177]
    q_pro = [0, 130, 156, 186, 500]
    q_i_lig = 2
    q_i_pro = 2
    num_clusters = [q_lig[q_i_lig], q_pro[q_i_pro]]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(save_dir, exist_ok=True)

    train_splits = load_split_txt(train_split_file)
    val_splits = load_split_txt(val_split_file)
    test_splits = load_split_txt(test_split_file)

    debut_train = time()
    for split_id in range(5):
        if log_callback:
            log_callback.info("\n==============================")
            log_callback.info(f"        SPLIT {split_id:02d}")
            log_callback.info("==============================")

        seed_everything(split_id)

        train_ids = train_splits[split_id]
        val_ids = val_splits[split_id]
        test_ids = test_splits[split_id]

        train_set = URVGraphDataset(graph_dir, train_ids)
        val_set = URVGraphDataset(graph_dir, val_ids)
        test_set = URVGraphDataset(graph_dir, test_ids)

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        all_train_y = []

        for data in train_loader:
            all_train_y.append(data.y)

        all_train_y = torch.cat(all_train_y)

        y_mean = all_train_y.mean().to(device)
        y_std = all_train_y.std().to(device)
        if log_callback:
            log_callback.info("Target mean:", y_mean.item())
            log_callback.info("Target std:", y_std.item())

            log_callback.info(f"Train samples: {len(train_set)}")
            log_callback.info(f"Test samples: {len(test_set)}")
            log_callback.info(f"Validation samples: {len(val_set)}")

        # ---------------- Model ----------------
        model = CheapNet(node_dim, hidden_dim, drop_rate, num_clusters).to(device)
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
        patience_counter = 0

        split_save_dir = os.path.join(save_dir, f"split_{split_id:02d}")
        os.makedirs(split_save_dir, exist_ok=True)

        # ---------------- Training loop ----------------
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            for data in train_loader:
                data = data.to(device)

                pred = model(data)
                y = (data.y - y_mean) / y_std
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * data.y.size(0)

            train_rmse = np.sqrt(epoch_loss / len(train_set))
            test_rmse, test_pr = val(model, test_loader, device)
            val_rmse, _ = val(model, val_loader, device)
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
                    {
                        "model_state_dict": model.state_dict(),
                        "y_mean": y_mean,
                        "y_std": y_std,
                    },
                    os.path.join(split_save_dir, "best_model.pt"),
                )
                if log_callback:
                    log_callback.info(">>> Nuevo mejor modelo guardado")

            else:
                patience_counter += 1

            if patience_counter >= patience:
                if log_callback:
                    log_callback.info(">>> Early stopping activado")
                break

        if log_callback:
            log_callback.info(f"Best RMSE split {split_id:02d}: {best_rmse:.4f}")
    end_train = time()
    train_time = end_train - debut_train

    write_hyperparameter_in_a_file(hidden_dim, node_dim, drop_rate, epochs, batch_size, lr, weight_decay, patience,
                                   "cheapnet_hyperparameter.txt")
    if log_callback:
        log_callback.info(f"\n it took {train_time:.2f} seconds to train all splits.")
        log_callback.info("\n Training completed for CheapNet.")
        log_callback.info("Hyperparameters saved in cheapnet_hyperparameter.txt")
