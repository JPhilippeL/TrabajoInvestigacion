"""
@file egnn_trainer.py
@author Mohamed EL BOUKHIARI
@brief Training pipeline for the EGNN model.
@details
This file is adapted from 04_c_Train_EGNN.py.
It exposes a callable train(...) function so that the training process
can later be triggered from the GUI through workers.py.
"""

from __future__ import annotations

import os
import ast
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

from .egnn_model import EGNN


# ============================================================
# UTILS
# ============================================================

def load_split_txt(path: str):
    """
    @brief Load split indices from a text file.
    @param path Path to the split file.
    @return Parsed split structure.
    """
    with open(path, "r") as f:
        return ast.literal_eval(f.read())


def seed_everything(seed: int):
    """
    @brief Set all random seeds for reproducibility.
    @param seed Random seed.
    @return None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def val(model, dataloader, device):
    """
    @brief Run validation.
    @param model EGNN model.
    @param dataloader Validation dataloader.
    @param device Computation device.
    @return Validation metrics/results according to original implementation.
    """

    # TODO:
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

    rmse = np.sqrt(((pred - label) ** 2).mean())
    pearson = np.corrcoef(pred, label)[0, 1]

    model.train()
    return rmse, pearson


# ============================================================
# DATASET
# ============================================================

class URVGraphDataset(Dataset):
    """
    @brief Dataset wrapper for generated EGNN graphs.
    @param graph_dir Directory containing graph files.
    @param pdb_ids List of graph identifiers.
    """

    def __init__(self, graph_dir, pdb_ids):
        self.graph_dir = graph_dir
        self.pdb_ids = pdb_ids


    def __len__(self):
        """
        @brief Return dataset size.
        @return Number of graphs.
        """
        return len(self.pdb_ids)

    def __getitem__(self, idx):
        """
        @brief Load one graph by index.
        @param idx Graph index.
        @return PyG Data object.
        """
        pdb_id = self.pdb_ids[idx]
        path = os.path.join(self.graph_dir, f"{pdb_id}.pt")
        return torch.load(path)


# ============================================================
# TRAIN
# ============================================================

def train(
    graphs_dir: str,
    train_split_file: str,
    val_split_file: str,
    test_split_file: str,
    output_base: str,
    batch_size: int = 4,
    epochs: int = 50,
    patience: int = 10,
    lr: float = 1e-4,
    hidden_dim: int = 64,
    device: str | None = None,
    seed: int = 42,
):
    """
    @brief Train EGNN on the predefined dataset splits.
    @param graphs_dir Directory containing generated graphs.
    @param train_split_file Path to train_index_folder.txt.
    @param val_split_file Path to valid_index_folder.txt.
    @param test_split_file Path to test_index_folder.txt.
    @param output_base Output directory for trained models.
    @param batch_size Batch size.
    @param epochs Number of epochs.
    @param patience Early stopping patience.
    @param lr Learning rate.
    @param hidden_dim Hidden dimension of the model.
    @param device Device to use.
    @param seed Random seed.
    @return Directory containing the saved trained models.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_base, exist_ok=True)

    train_splits = load_split_txt(train_split_file)
    val_splits = load_split_txt(val_split_file)
    test_splits = load_split_txt(test_split_file)

    for split_id in range(5):
        print(f"\n==============================")
        print(f"        SPLIT {split_id:02d}")
        print(f"==============================")

        seed_everything(seed + split_id)

        train_ids = train_splits[split_id]
        val_ids = val_splits[split_id]
        test_ids = test_splits[split_id]

        train_set = URVGraphDataset(graphs_dir, train_ids)
        test_set = URVGraphDataset(graphs_dir, test_ids)
        val_set = URVGraphDataset(graphs_dir, val_ids)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=batch_size)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        print("Train samples:", len(train_set))
        print("Test samples:", len(test_set))
        print("Validation samples:", len(val_set))

        model = EGNN(hidden_dim=hidden_dim).to(device)

        print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_rmse = float("inf")
        patience_counter = 0

        split_save_dir = os.path.join(output_base, f"split_{split_id:02d}")
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
            val_rmse, _ = val(model, val_loader, device)

            print(
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
                    os.path.join(split_save_dir, "best_model.pt")
                )
                print(">>> Nuevo mejor modelo guardado")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(">>> Early stopping activado")
                break

        print(f"Best RMSE split {split_id:02d}: {best_rmse:.4f}")

    print("\nEntrenamiento completado para todos los splits.")
    return output_base

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    MODULE_ROOT = os.path.dirname(PROJECT_ROOT)

    train(
        graphs_dir=os.path.join(MODULE_ROOT, "Graphs_EGNN"),
        dataset_dir=os.path.join(MODULE_ROOT, "MPro-URV_Version2"),
        models_dir=os.path.join(MODULE_ROOT, "Models_EGNN"),
    )
