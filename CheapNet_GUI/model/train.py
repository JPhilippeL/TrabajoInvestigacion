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

    split_best_val_rmses = []
    split_test_rmses = []
    split_test_pearsons = []
    for split_id in range(len(train_splits)):
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
            train_set, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        all_train_y = []

        for data in train_loader:
            all_train_y.append(data.y.view(-1))

        all_train_y = torch.cat(all_train_y).to(device)

        y_mean = all_train_y.mean()
        y_std = all_train_y.std()
        if log_callback:
            log_callback.info(f"Target mean:{y_mean.item()}")
            log_callback.info(f"Target std: {y_std.item()}")

            log_callback.info(f"Train samples: {len(train_set)}")
            log_callback.info(f"Test samples: {len(test_set)}")
            log_callback.info(f"Validation samples: {len(val_set)}")

        model = CheapNet(node_dim, hidden_dim, drop_rate, num_clusters).to(device)
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
                y = (target - y_mean) / y_std
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                current_batch_size = target.size(0)
                epoch_loss += loss.item() * current_batch_size
                n_train += current_batch_size

            train_rmse = np.sqrt(epoch_loss / n_train)
            test_rmse, test_pr = val(model, test_loader, device, y_mean, y_std)
            val_rmse, _ = val(model, val_loader, device, y_mean, y_std)
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
                        "split_id": split_id,
                        "best_epoch": epoch,
                        "best_val_rmse": best_rmse,
                        "y_mean": y_mean,
                        "y_std": y_std,
                        "config": {
                            "node_dim": node_dim,
                            "hidden_dim": hidden_dim,
                            "drop_out": drop_rate,
                            "batch_size": batch_size,
                            "lr": lr,
                            "weight_decay": weight_decay,
                            "epochs": epochs,
                            "patience": patience,
                        },
                    },
                    best_model_path,
                )
                if log_callback:
                    log_callback.info(">>> Better model stored")

            else:
                patience_counter += 1

            if patience_counter >= patience:
                if log_callback:
                    log_callback.info(">>> Early stopping activado")
                break

        checkpoint = torch.load(
            best_model_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        y_mean = checkpoint["y_mean"].to(device)
        y_std = checkpoint["y_std"].to(device)

        best_test_rmse, best_test_pr = val(
            model,
            test_loader,
            device,
            y_mean,
            y_std,
        )

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

    end_train = time()
    train_time = end_train - debut_train

    mean_val_rmse = float(np.mean(split_best_val_rmses))
    mean_test_rmse = float(np.mean(split_test_rmses))
    mean_test_pearson = float(np.mean(split_test_pearsons))

    std_val_rmse = float(np.std(split_best_val_rmses))
    std_test_rmse = float(np.std(split_test_rmses))
    std_test_pearson = float(np.std(split_test_pearsons))

    hyperparameter_path = os.path.join(save_dir, "cheapnet_hyperparameter.txt")

    write_hyperparameter_in_a_file(
        hidden_dim,
        node_dim,
        drop_rate,
        epochs,
        batch_size,
        lr,
        weight_decay,
        patience,
        hyperparameter_path,
    )

    if log_callback:
        log_callback.info(f"\nIt took {train_time:.2f} seconds to train all splits.")
        log_callback.info("\nTraining completed for CheapNet.")
        log_callback.info(f"Mean Val RMSE: {mean_val_rmse:.4f}")
        log_callback.info(f"Mean Test RMSE: {mean_test_rmse:.4f}")
        log_callback.info(f"Mean Test Pearson: {mean_test_pearson:.4f}")
        log_callback.info(f"Hyperparameters saved in {hyperparameter_path}")

    return {
        "mean_val_rmse": mean_val_rmse,
        "std_val_rmse": std_val_rmse,
        "mean_test_rmse": mean_test_rmse,
        "std_test_rmse": std_test_rmse,
        "mean_test_pearson": mean_test_pearson,
        "std_test_pearson": std_test_pearson,
    }
