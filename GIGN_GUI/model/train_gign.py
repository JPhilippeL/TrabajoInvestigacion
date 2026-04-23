import os
import ast
import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import mean_squared_error
from GIGN_GUI.model.GIGN_model import GIGN
from GNNs.explainers.graph_explainer_onehot import obtener_graph_explainer


# =========================================================
# Utils
# =========================================================


def load_split_txt(path):
    with open(path, "r") as f:
        return ast.literal_eval(f.read())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def val(model, dataloader, device):
    model.eval()
    pred_list, label_list = [], []

    for data in dataloader:
        data = data.to(device)
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


# =========================================================
# Dataset
# =========================================================


class URVGraphDataset(Dataset):
    def __init__(self, graph_dir, pdb_ids):
        self.graph_dir = graph_dir
        self.pdb_ids = pdb_ids

    def __len__(self):
        return len(self.pdb_ids)

    def __getitem__(self, idx):
        pdb_id = self.pdb_ids[idx]
        path = os.path.join(self.graph_dir, f"{pdb_id}.pt")
        data = torch.load(path, weights_only=False)
        return data


def write_hyperparameter_into_a_file(epochs, node_dim, hidden_dim, drop_out, batch_size, lr, weight_decay, patience,
                                     output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True) if os.path.dirname(output_file) else None
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Hyperparameters:\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"node_dim: {node_dim}\n")
        f.write(f"hidden_dim: {hidden_dim}\n")
        f.write(f"drop_out: {drop_out}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"weight_decay: {weight_decay}\n")
        f.write(f"patience: {patience}\n")


# =========================================================
# Main
# =========================================================
def train_gign(train_file, test_file, val_file, epochs, seed, node_dim, hidden_dim, batch_size, lr, weight_decay,
               patience, drop_out, save_dir, log_callback, graph_dir):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)

    train_splits = load_split_txt(train_file)
    val_splits = load_split_txt(val_file)
    test_splits = load_split_txt(test_file)

    for split_id in range(5):
        if log_callback:
            log_callback.info(f"SPLIT {split_id}")

        # Reproducibilidad por split
        seed_everything(split_id + seed)

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
        if log_callback:
            log_callback.info(f"Train samples: {len(train_set)}")
            log_callback.info(f"Test samples: {len(test_set)}")
            log_callback.info(f"Validation samples: {len(val_set)}")

        # ---------------- Model ----------------
        model = GIGN(node_dim, hidden_dim, drop_out).to(DEVICE)
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

        # ---------------- Training loop ----------------
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            for data in train_loader:
                data = data.to(DEVICE)

                pred = model(data)
                loss = criterion(pred, data.y.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * data.y.size(0)

            train_rmse = np.sqrt(epoch_loss / len(train_set))
            test_rmse, test_pr = val(model, test_loader, DEVICE)
            val_rmse, val_pr = val(model, val_loader, DEVICE)

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
                    model.state_dict(),
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

    if log_callback:
        log_callback.info("\nEntrenamiento completado para todos los splits.")

    write_hyperparameter_into_a_file(epochs, node_dim, hidden_dim, drop_out, batch_size, lr, weight_decay, patience,
                                     os.path.join(save_dir, "hyperparameters.txt"))
