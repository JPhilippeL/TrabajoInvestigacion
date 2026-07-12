"""
@file widedta_trainer.py
@author Mohamed EL BOUKHIARI
@brief Clean training pipeline for the WideDTA module.
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

from WideDTA.Core.widedta_audit import audit_dataset_splits
from WideDTA.data import WideDTADataset
from WideDTA.model import WideCNN


MODULE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class RMSLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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


def get_dataset_paths(dataset_name: str) -> tuple[str, str, str, str]:
    dataset_name = dataset_name.lower().strip()
    supported_datasets = {"davis", "kiba", "mpro_urv"}

    if dataset_name not in supported_datasets:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Expected one of: {sorted(supported_datasets)}."
        )

    dataset_dir = os.path.join(MODULE_ROOT, "data", dataset_name)

    ligand_path = os.path.join(dataset_dir, "ligands_can.txt")
    protein_path = os.path.join(dataset_dir, "proteins.txt")
    motif_path = os.path.join(dataset_dir, "motif2.txt")
    affinity_path = os.path.join(dataset_dir, "Y")

    for path in (ligand_path, protein_path, motif_path, affinity_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required WideDTA dataset file not found: {path}")

    return ligand_path, protein_path, motif_path, affinity_path


def get_dataset_fold_paths(dataset_name: str) -> tuple[str, str, str]:
    dataset_dir = os.path.join(MODULE_ROOT, "data", dataset_name.lower().strip())
    folds_dir = os.path.join(dataset_dir, "folds")

    train_fold_path = os.path.join(folds_dir, "train_fold_setting1.txt")
    valid_fold_path = os.path.join(folds_dir, "valid_fold_setting1.txt")
    test_fold_path = os.path.join(folds_dir, "test_fold_setting1.txt")

    for path in (train_fold_path, valid_fold_path, test_fold_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required WideDTA fold file not found: {path}")

    return train_fold_path, valid_fold_path, test_fold_path


def dataset_has_fold_files(dataset_name: str) -> bool:
    dataset_dir = os.path.join(MODULE_ROOT, "data", dataset_name.lower().strip())
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

    parsed = ast.literal_eval(content)

    if isinstance(parsed, list) and parsed and all(isinstance(item, int) for item in parsed):
        return [parsed]

    if isinstance(parsed, list) and parsed and all(isinstance(item, list) for item in parsed):
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
    dataset: WideDTADataset,
    batch_size: int,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_size = len(dataset)

    validate_indices(train_indices, dataset_size, "train")
    validate_indices(val_indices, dataset_size, "validation")
    validate_indices(test_indices, dataset_size, "test")

    if set(train_indices) & set(val_indices) or set(train_indices) & set(test_indices) or set(val_indices) & set(test_indices):
        raise ValueError("Dataset splits overlap.")

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices))
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices))

    return train_loader, val_loader, test_loader


def build_dataloaders_from_fold_files(
    dataset: WideDTADataset,
    dataset_name: str,
    batch_size: int,
    fold_index: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_fold_path, valid_fold_path, test_fold_path = get_dataset_fold_paths(dataset_name)

    train_folds = read_fold_file(train_fold_path)
    valid_folds = read_fold_file(valid_fold_path)
    test_folds = read_fold_file(test_fold_path)

    num_folds = min(len(train_folds), len(valid_folds), len(test_folds))

    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError(
            f"Invalid fold_index={fold_index}. Dataset '{dataset_name}' has {num_folds} fold(s)."
        )

    return build_dataloaders_from_indices(
        dataset=dataset,
        batch_size=batch_size,
        train_indices=train_folds[fold_index],
        val_indices=valid_folds[fold_index],
        test_indices=test_folds[fold_index],
    )


def build_dataloaders(
    dataset: WideDTADataset,
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

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    test_size = max(1, int(math.floor(test_split * dataset_size)))
    val_size = max(1, int(math.floor(val_split * dataset_size))) if val_split > 0 else 1

    if test_size + val_size >= dataset_size:
        raise ValueError(
            f"Dataset too small for val/test split. size={dataset_size}, val={val_size}, test={test_size}"
        )

    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    return build_dataloaders_from_indices(dataset, batch_size, train_indices, val_indices, test_indices)


def prepare_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features, target = batch
    ligand, protein, motif = features

    ligand = ligand.float().to(device)
    protein = protein.float().to(device)
    motif = motif.float().to(device)
    target = target.float().to(device).reshape(-1, 1)

    return protein, ligand, motif, target


def initialize_model_for_dataset(model: nn.Module, dataset: WideDTADataset, device: torch.device) -> None:
    """
    @brief Initialize LazyConv1d/LazyLinear layers using one zero batch matching dataset shapes.
    """
    shapes = dataset.input_shapes()

    ligand = torch.zeros((1, *shapes["ligand"]), dtype=torch.float32, device=device)
    protein = torch.zeros((1, *shapes["protein"]), dtype=torch.float32, device=device)
    motif = torch.zeros((1, *shapes["motif"]), dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        model(protein, ligand, motif)


def regression_metrics(predictions: list[float], targets: list[float]) -> Dict[str, float]:
    pred = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(targets, dtype=np.float64)

    if pred.size == 0:
        return {"RMSE": float("nan"), "MSE": float("nan"), "MAE": float("nan"), "Pearson": float("nan")}

    mse = float(np.mean((pred - true) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(pred - true)))

    if pred.size < 2 or np.std(pred) == 0 or np.std(true) == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(pred, true)[0, 1])

    return {"RMSE": rmse, "MSE": mse, "MAE": mae, "Pearson": pearson}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    losses: list[float] = []
    predictions: list[float] = []
    targets: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            protein, ligand, motif, target = prepare_batch(batch, device)
            output = model(protein, ligand, motif)
            target = target.reshape_as(output)

            loss = criterion(output, target)
            losses.append(float(loss.item()))

            predictions.extend(output.detach().cpu().numpy().reshape(-1).tolist())
            targets.extend(target.detach().cpu().numpy().reshape(-1).tolist())

    metrics = regression_metrics(predictions, targets)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def save_checkpoint(
    model: WideCNN,
    checkpoint_path: str,
    dataset: WideDTADataset,
    model_kwargs: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "input_shapes": dataset.input_shapes(),
            "metadata": metadata,
        },
        checkpoint_path,
    )


def load_checkpoint(checkpoint_path: str, dataset: WideDTADataset, device: torch.device) -> WideCNN:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    model_kwargs = checkpoint.get("model_kwargs", {})
    model = WideCNN(**model_kwargs).to(device)
    initialize_model_for_dataset(model, dataset, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def train(
    dataset_name: str = "mpro_urv",
    output_base: str | None = None,
    batch_size: int = 1,
    epochs: int = 3,
    lr: float = 0.003,
    dropout: float = 0.3,
    device: str | None = "auto",
    seed: int = 42,
    val_split: float = 0.1,
    test_split: float = 0.2,
    max_train_batches: int | None = None,
    fold_index: int = 0,
    use_dataset_folds: bool = True,
) -> Dict[str, Any]:
    """
    @brief Train WideDTA cleanly without hardcoded Davis/KIBA tensor shapes.
    @return Dictionary containing metrics and checkpoint path.
    """
    set_seed(seed)
    if max_train_batches == 0:
        max_train_batches = None

    dataset_name = dataset_name.lower().strip()
    torch_device = resolve_device(device)

    if output_base is None:
        output_base = os.path.join(MODULE_ROOT, "results", "widedta_runs")

    os.makedirs(output_base, exist_ok=True)

    ligand_path, protein_path, motif_path, affinity_path = get_dataset_paths(dataset_name)
    dataset = WideDTADataset(ligand_path, protein_path, motif_path, affinity_path)
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

    model_kwargs = {"dropout": dropout}
    model = WideCNN(**model_kwargs).to(torch_device)
    initialize_model_for_dataset(model, dataset, torch_device)

    criterion = RMSLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_rmse = float("inf")
    best_checkpoint_path = os.path.join(output_base, "widedta_best.pt")
    checkpoint_saved = False

    print(
        f"[WideDTA] dataset={dataset_name} samples={len(dataset)} "
        f"input_shapes={dataset.input_shapes()} split_mode={split_mode} "
        f"fold_index={fold_index if split_mode == 'dataset_folds' else 'NA'} "
        f"batch_size={batch_size} epochs={epochs} lr={lr} dropout={dropout} device={torch_device}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []

        for batch_index, batch in enumerate(train_loader, start=1):
            protein, ligand, motif, target = prepare_batch(batch, torch_device)
            output = model(protein, ligand, motif)
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
            f"[WideDTA] epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_rmse={val_rmse:.6f} val_pearson={val_metrics['Pearson']:.6f}"
        )

        if math.isfinite(val_rmse) and val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            save_checkpoint(
                model=model,
                checkpoint_path=best_checkpoint_path,
                dataset=dataset,
                model_kwargs=model_kwargs,
                metadata={
                    "dataset": dataset_name,
                    "epoch": epoch,
                    "lr": lr,
                    "batch_size": batch_size,
                    "dropout": dropout,
                    "seed": seed,
                    "split_mode": split_mode,
                },
            )
            checkpoint_saved = True

    if not checkpoint_saved:
        save_checkpoint(
            model=model,
            checkpoint_path=best_checkpoint_path,
            dataset=dataset,
            model_kwargs=model_kwargs,
            metadata={
                "dataset": dataset_name,
                "epoch": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "dropout": dropout,
                "seed": seed,
                "split_mode": split_mode,
            },
        )

    best_model = load_checkpoint(best_checkpoint_path, dataset, torch_device)

    train_metrics = evaluate(best_model, train_loader, criterion, torch_device)
    val_metrics = evaluate(best_model, val_loader, criterion, torch_device)
    test_metrics = evaluate(best_model, test_loader, criterion, torch_device)

    return {
        "dataset": dataset_name,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "dropout": dropout,
        "device": str(torch_device),
        "seed": seed,
        "split_mode": split_mode,
        "fold_index": fold_index if split_mode == "dataset_folds" else None,
        "input_shapes": dataset.input_shapes(),
        "train_rmse": train_metrics["RMSE"],
        "train_pearson": train_metrics["Pearson"],
        "val_rmse": val_metrics["RMSE"],
        "val_pearson": val_metrics["Pearson"],
        "test_rmse": test_metrics["RMSE"],
        "test_pearson": test_metrics["Pearson"],
        "test_mse": test_metrics["MSE"],
        "test_mae": test_metrics["MAE"],
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
