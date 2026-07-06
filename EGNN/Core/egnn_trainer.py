"""
@file egnn_trainer.py
@author Mohamed EL BOUKHIARI
@brief Training pipeline for the EGNN model.
@details
This file is adapted from 04_c_Train_EGNN.py.
It exposes a callable train(...) function so that the training process can be
triggered from the GUI through workers.py.
"""

from __future__ import annotations

import ast
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

from .egnn_model import EGNN


def load_split_txt(path: str):
    """
    @brief Load split indices from a text file.
    @param path Path to the split file.
    @return Parsed split structure.
    """
    with open(path, "r", encoding="utf-8") as f:
        return ast.literal_eval(f.read())


def seed_everything(seed: int) -> None:
    """
    @brief Set all random seeds for reproducibility.
    @param seed Random seed.
    @return None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None) -> str:
    """
    @brief Resolve the requested device into a valid torch device string.
    @param device Requested device. Use None or "auto" for automatic selection.
    @return Device string.
    """
    if device is None or device == "" or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA was requested but is not available. Falling back to CPU.")
        return "cpu"

    return device


def safe_torch_load(path: str) -> Any:
    """
    @brief Load PyTorch objects while remaining compatible with PyTorch versions
           that introduced weights_only=True as a safer default.
    @param path File path.
    @return Loaded object.
    """
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def evaluate_regression(model: EGNN, dataloader: DataLoader, device: str) -> tuple[float, float]:
    """
    @brief Evaluate RMSE and Pearson correlation on a dataloader.
    @param model EGNN model.
    @param dataloader Evaluation dataloader.
    @param device Computation device.
    @return Tuple (RMSE, Pearson).
    """
    model.eval()
    pred_list = []
    label_list = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            pred = model(data)

            pred_list.append(pred.cpu().numpy())
            label_list.append(data.y.view(-1).cpu().numpy())

    if not pred_list:
        model.train()
        return float("nan"), float("nan")

    preds = np.concatenate(pred_list)
    labels = np.concatenate(label_list)

    rmse = float(np.sqrt(((preds - labels) ** 2).mean()))
    pearson = float(np.corrcoef(preds, labels)[0, 1]) if len(labels) > 1 else float("nan")

    model.train()
    return rmse, pearson


class URVGraphDataset(Dataset):
    """
    @brief Dataset wrapper for generated EGNN graphs.
    @param graphs_dir Directory containing graph files.
    @param pdb_ids List of graph identifiers.
    """

    def __init__(self, graphs_dir: str, pdb_ids: list[str]):
        self.graphs_dir = graphs_dir
        self.pdb_ids = pdb_ids

    def __len__(self) -> int:
        return len(self.pdb_ids)

    def __getitem__(self, idx: int):
        pdb_id = self.pdb_ids[idx]
        path = os.path.join(self.graphs_dir, f"{pdb_id}.pt")
        return safe_torch_load(path)


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
    @brief Train EGNN on predefined dataset splits.
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
    @param device Device to use. Use "auto" for automatic selection.
    @param seed Random seed.
    @return Directory containing the saved trained models.
    """
    device = resolve_device(device)

    os.makedirs(output_base, exist_ok=True)

    train_splits = load_split_txt(train_split_file)
    val_splits = load_split_txt(val_split_file)
    test_splits = load_split_txt(test_split_file)

    for split_id in range(5):
        print("\n==============================")
        print(f"        SPLIT {split_id:02d}")
        print("==============================")

        seed_everything(seed + split_id)

        train_ids = train_splits[split_id]
        val_ids = val_splits[split_id]
        test_ids = test_splits[split_id]

        train_set = URVGraphDataset(graphs_dir, train_ids)
        val_set = URVGraphDataset(graphs_dir, val_ids)
        test_set = URVGraphDataset(graphs_dir, test_ids)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        print("Train samples:", len(train_set))
        print("Validation samples:", len(val_set))
        print("Test samples:", len(test_set))
        print("Device:", device)

        model = EGNN(hidden_dim=hidden_dim).to(device)

        print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_rmse = float("inf")
        patience_counter = 0

        split_save_dir = os.path.join(output_base, f"split_{split_id:02d}")
        os.makedirs(split_save_dir, exist_ok=True)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            sample_count = 0

            for data in train_loader:
                data = data.to(device)

                pred = model(data)
                target = data.y.view(-1)
                loss = criterion(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_count = target.size(0)
                epoch_loss += loss.item() * batch_count
                sample_count += batch_count

            train_rmse = float(np.sqrt(epoch_loss / sample_count)) if sample_count else float("nan")
            val_rmse, _ = evaluate_regression(model, val_loader, device)
            test_rmse, test_pearson = evaluate_regression(model, test_loader, device)

            print(
                f"Epoch {epoch:03d} | "
                f"Train RMSE: {train_rmse:.4f} | "
                f"Validation RMSE: {val_rmse:.4f} | "
                f"Test RMSE: {test_rmse:.4f} | "
                f"Test Pearson: {test_pearson:.4f}"
            )

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0

                torch.save(
                    model.state_dict(),
                    os.path.join(split_save_dir, "best_model.pt"),
                )
                print(">>> New best model saved.")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(">>> Early stopping triggered.")
                break

        print(f"Best validation RMSE for split {split_id:02d}: {best_val_rmse:.4f}")

    print("\nTraining completed for all splits.")
    return output_base


if __name__ == "__main__":
    from EGNN.utils.constants import (
        DEFAULT_GRAPHS_DIR,
        DEFAULT_MODELS_DIR,
        DEFAULT_TRAIN_SPLIT_FILE,
        DEFAULT_VAL_SPLIT_FILE,
        DEFAULT_TEST_SPLIT_FILE,
    )

    train(
        graphs_dir=DEFAULT_GRAPHS_DIR,
        train_split_file=DEFAULT_TRAIN_SPLIT_FILE,
        val_split_file=DEFAULT_VAL_SPLIT_FILE,
        test_split_file=DEFAULT_TEST_SPLIT_FILE,
        output_base=DEFAULT_MODELS_DIR,
    )
