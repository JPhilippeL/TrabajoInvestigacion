"""
@file deepdta_trainer.py
@author Mohamed EL BOUKHIARI
@brief Clean training pipeline for the DeepDTA module.
"""

from __future__ import annotations

import ast
import math
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from DeepDTA.Core.deepdta_audit import audit_dataset_splits
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

    supported_datasets = {"davis", "kiba", "mpro_urv"}

    if dataset_name not in supported_datasets:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Expected one of: {sorted(supported_datasets)}."
        )

    dataset_dir = os.path.join(MODULE_ROOT, "data", dataset_name)

    ligand_path = os.path.join(dataset_dir, "ligands_can.txt")
    protein_path = os.path.join(dataset_dir, "proteins.txt")
    affinity_path = os.path.join(dataset_dir, "Y")

    for path in (ligand_path, protein_path, affinity_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required DeepDTA dataset file not found: {path}")

    return ligand_path, protein_path, affinity_path


def get_dataset_fold_paths(dataset_name: str) -> tuple[str, str, str]:
    dataset_name = dataset_name.lower().strip()

    dataset_dir = os.path.join(MODULE_ROOT, "data", dataset_name)
    folds_dir = os.path.join(dataset_dir, "folds")

    train_fold_path = os.path.join(folds_dir, "train_fold_setting1.txt")
    valid_fold_path = os.path.join(folds_dir, "valid_fold_setting1.txt")
    test_fold_path = os.path.join(folds_dir, "test_fold_setting1.txt")

    for path in (train_fold_path, valid_fold_path, test_fold_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required DeepDTA fold file not found: {path}")

    return train_fold_path, valid_fold_path, test_fold_path


def dataset_has_fold_files(dataset_name: str) -> bool:
    dataset_name = dataset_name.lower().strip()

    dataset_dir = os.path.join(MODULE_ROOT, "data", dataset_name)
    folds_dir = os.path.join(dataset_dir, "folds")

    required_files = [
        os.path.join(folds_dir, "train_fold_setting1.txt"),
        os.path.join(folds_dir, "valid_fold_setting1.txt"),
        os.path.join(folds_dir, "test_fold_setting1.txt"),
    ]

    return all(os.path.exists(path) for path in required_files)


def read_fold_file(path: str) -> list[list[int]]:
    with open(path, "r", encoding="utf-8") as file:
        content = file.read().strip()

    if not content:
        raise ValueError(f"Fold file is empty: {path}")

    try:
        parsed = ast.literal_eval(content)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse fold file: {path}") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"Fold file must contain a list: {path}")

    if parsed and all(isinstance(item, int) for item in parsed):
        return [parsed]

    if parsed and all(isinstance(item, list) for item in parsed):
        for fold in parsed:
            if not all(isinstance(index, int) for index in fold):
                raise ValueError(f"Fold file contains non-integer indices: {path}")
        return parsed

    raise ValueError(f"Unsupported fold file structure: {path}")


def validate_indices(indices: list[int], dataset_size: int, split_name: str) -> None:
    if not indices:
        raise ValueError(f"{split_name} split is empty.")

    invalid_indices = [index for index in indices if index < 0 or index >= dataset_size]

    if invalid_indices:
        raise ValueError(
            f"{split_name} split contains invalid indices. "
            f"Dataset size={dataset_size}, invalid examples={invalid_indices[:10]}"
        )


def build_dataloaders_from_indices(
    dataset: NumbersDataset,
    batch_size: int,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_size = len(dataset)

    validate_indices(train_indices, dataset_size, "train")
    validate_indices(val_indices, dataset_size, "validation")
    validate_indices(test_indices, dataset_size, "test")

    train_overlap = set(train_indices) & set(val_indices)
    test_overlap = set(train_indices) & set(test_indices)
    val_test_overlap = set(val_indices) & set(test_indices)

    if train_overlap or test_overlap or val_test_overlap:
        raise ValueError(
            "Dataset splits overlap. "
            f"train/val={len(train_overlap)}, "
            f"train/test={len(test_overlap)}, "
            f"val/test={len(val_test_overlap)}"
        )

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader


def build_dataloaders_from_fold_files(
    dataset: NumbersDataset,
    dataset_name: str,
    batch_size: int,
    fold_index: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_fold_path, valid_fold_path, test_fold_path = get_dataset_fold_paths(dataset_name)

    train_folds = read_fold_file(train_fold_path)
    valid_folds = read_fold_file(valid_fold_path)
    test_folds = read_fold_file(test_fold_path)

    num_folds = min(len(train_folds), len(valid_folds), len(test_folds))

    if num_folds == 0:
        raise ValueError(f"No usable folds found for dataset '{dataset_name}'.")

    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError(
            f"Invalid fold_index={fold_index}. "
            f"Dataset '{dataset_name}' has {num_folds} fold(s), valid range: 0..{num_folds - 1}."
        )

    train_indices = train_folds[fold_index]
    val_indices = valid_folds[fold_index]
    test_indices = test_folds[fold_index]

    return build_dataloaders_from_indices(
        dataset=dataset,
        batch_size=batch_size,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )


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

    return build_dataloaders_from_indices(
        dataset=dataset,
        batch_size=batch_size,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )


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


def load_saved_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    try:
        model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(checkpoint_path, map_location=device)

    model.to(device)
    return model


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
    fold_index: int = 0,
    use_dataset_folds: bool = True,
) -> Dict[str, Any]:
    """
    @brief Train DeepDTA cleanly without executing the old train.py script.
    @return Dictionary containing metrics and checkpoint path.
    """
    set_seed(seed)
    if max_train_batches == 0:
        max_train_batches = None

    dataset_name = dataset_name.lower().strip()
    torch_device = resolve_device(device)

    if output_base is None:
        output_base = os.path.join(MODULE_ROOT, "results", "deepdta_runs")

    os.makedirs(output_base, exist_ok=True)

    ligand_path, protein_path, affinity_path = get_dataset_paths(dataset_name)
    dataset = NumbersDataset(ligand_path, protein_path, affinity_path)
    split_audit = audit_dataset_splits(
        dataset_name=dataset_name,
        fold_index=fold_index,
        use_dataset_folds=use_dataset_folds,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        output_dir=output_base,
    )

    if use_dataset_folds and dataset_has_fold_files(dataset_name):
        split_mode = "dataset_folds"
        train_loader, val_loader, test_loader = build_dataloaders_from_fold_files(
            dataset=dataset,
            dataset_name=dataset_name,
            batch_size=batch_size,
            fold_index=fold_index,
        )
    else:
        split_mode = "random"
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
    checkpoint_saved = False

    print(
        f"[DeepDTA] dataset={dataset_name} "
        f"samples={len(dataset)} "
        f"split_mode={split_mode} "
        f"fold_index={fold_index if split_mode == 'dataset_folds' else 'NA'} "
        f"batch_size={batch_size} "
        f"epochs={epochs} "
        f"lr={lr} "
        f"device={torch_device}"
    )

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

        if math.isfinite(val_rmse) and val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model, best_checkpoint_path)
            checkpoint_saved = True

    if not checkpoint_saved:
        torch.save(model, best_checkpoint_path)

    best_model = load_saved_model(best_checkpoint_path, torch_device)

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
        "split_mode": split_mode,
        "fold_index": fold_index if split_mode == "dataset_folds" else None,
        "train_rmse": train_metrics["RMSE"],
        "train_pearson": train_metrics["Pearson"],
        "val_rmse": val_metrics["RMSE"],
        "val_pearson": val_metrics["Pearson"],
        "test_rmse": test_metrics["RMSE"],
        "test_pearson": test_metrics["Pearson"],
        "checkpoint_path": best_checkpoint_path,
        "output_base": output_base,
        "split_audit_path": split_audit.get("audit_path"),
        "warnings": split_audit.get("warnings", []) + (
            ["DEBUG RUN - metrics are not scientifically valid."]
            if max_train_batches is not None
            else []
        ),
        "debug_run": max_train_batches is not None,
    }
