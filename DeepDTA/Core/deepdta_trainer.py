"""
@file deepdta_trainer.py
@author Mohamed EL BOUKHIARI
@brief Clean training pipeline for the DeepDTA module.
"""

from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from DeepDTA.data import NumbersDataset
from DeepDTA.model import CNNcom


MODULE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class RMSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[str]) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def get_dataset_paths(dataset_name: str) -> tuple[str, str, str]:
    dataset_name = dataset_name.lower().strip()

    if dataset_name not in {"davis", "kiba"}:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Expected 'davis' or 'kiba'."
        )

    dataset_dir = os.path.join(MODULE_ROOT, dataset_name)

    ligand_path = os.path.join(dataset_dir, "ligands_can.txt")
    protein_path = os.path.join(dataset_dir, "proteins.txt")
    affinity_path = os.path.join(dataset_dir, "Y")

    for path in (ligand_path, protein_path, affinity_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required DeepDTA dataset file not found: {path}")

    return ligand_path, protein_path, affinity_path


def build_dataloaders(
    dataset: NumbersDataset,
    batch_size: int,
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if not 0.0 < test_split < 1.0:
        raise ValueError("test_split must be between 0 and 1.")

    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1.")

    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be lower than 1.")

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    np.random.seed(seed)
    np.random.shuffle(indices)

    test_size = int(math.floor(test_split * dataset_size))
    val_size = int(math.floor(val_split * dataset_size))

    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader


def prepare_batch(
    batch,
    device: torch.device,
    ligand_channels: int = 62,
    ligand_length: int = 50,
    protein_channels: int = 25,
    protein_length: int = 600,
):
    features, target = batch
    ligand, protein = features

    ligand = ligand.float().to(device)
    protein = protein.float().to(device)
    target = target.float().to(device)

    current_batch_size = ligand.shape[0]

    ligand = ligand.reshape(current_batch_size, ligand_channels, ligand_length)
    protein = protein.reshape(current_batch_size, protein_channels, protein_length)

    return ligand, protein, target


def regression_metrics(predictions: list[float], targets: list[float]) -> Dict[str, float]:
    pred = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(targets, dtype=np.float64)

    if pred.size == 0:
        return {
            "RMSE": float("nan"),
            "Pearson": float("nan"),
        }

    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))

    if pred.size < 2 or np.std(pred) == 0 or np.std(true) == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(pred, true)[0, 1])

    return {
        "RMSE": rmse,
        "Pearson": pearson,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    losses = []
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            ligand, protein, target = prepare_batch(batch, device)

            output = model(ligand, protein)
            target = target.reshape_as(output)

            loss = criterion(output, target)
            losses.append(float(loss.item()))

            predictions.extend(output.detach().cpu().numpy().reshape(-1).tolist())
            targets.extend(target.detach().cpu().numpy().reshape(-1).tolist())

    metrics = regression_metrics(predictions, targets)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")

    return metrics


def train(
    dataset_name: str = "davis",
    output_base: str | None = None,
    batch_size: int = 4,
    epochs: int = 3,
    lr: float = 0.003,
    device: str | None = "auto",
    seed: int = 42,
    val_split: float = 0.1,
    test_split: float = 0.2,
    max_train_batches: int | None = None,
) -> Dict[str, Any]:
    """
    @brief Train DeepDTA cleanly without executing the old train.py script.
    @return Dictionary containing metrics and checkpoint path.
    """
    set_seed(seed)

    torch_device = resolve_device(device)

    if output_base is None:
        output_base = os.path.join(MODULE_ROOT, "results", "deepdta_runs")

    os.makedirs(output_base, exist_ok=True)

    ligand_path, protein_path, affinity_path = get_dataset_paths(dataset_name)
    dataset = NumbersDataset(ligand_path, protein_path, affinity_path)

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset=dataset,
        batch_size=batch_size,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )

    model = CNNcom().to(torch_device)
    criterion = RMSLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_rmse = float("inf")
    best_checkpoint_path = os.path.join(output_base, "deepdta_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()

        train_losses = []

        for batch_index, batch in enumerate(train_loader, start=1):
            ligand, protein, target = prepare_batch(batch, torch_device)

            output = model(ligand, protein)
            target = target.reshape_as(output)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

            if max_train_batches is not None and batch_index >= max_train_batches:
                break

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_metrics = evaluate(model, val_loader, criterion, torch_device)

        val_rmse = val_metrics["RMSE"]

        print(
            f"[DeepDTA] epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_rmse={val_rmse:.6f} "
            f"val_pearson={val_metrics['Pearson']:.6f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model, best_checkpoint_path)

    best_model = torch.load(best_checkpoint_path, map_location=torch_device)
    best_model.to(torch_device)

    train_metrics = evaluate(best_model, train_loader, criterion, torch_device)
    val_metrics = evaluate(best_model, val_loader, criterion, torch_device)
    test_metrics = evaluate(best_model, test_loader, criterion, torch_device)

    return {
        "dataset": dataset_name,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "device": str(torch_device),
        "seed": seed,
        "train_rmse": train_metrics["RMSE"],
        "train_pearson": train_metrics["Pearson"],
        "val_rmse": val_metrics["RMSE"],
        "val_pearson": val_metrics["Pearson"],
        "test_rmse": test_metrics["RMSE"],
        "test_pearson": test_metrics["Pearson"],
        "checkpoint_path": best_checkpoint_path,
        "output_base": output_base,
    }
