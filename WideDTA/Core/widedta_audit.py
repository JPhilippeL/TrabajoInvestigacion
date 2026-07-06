"""
@file widedta_audit.py
@brief Split leakage diagnostics for WideDTA datasets.
"""

from __future__ import annotations

import ast
import json
import math
import os
import pickle
from datetime import datetime
from typing import Any

import numpy as np


MODULE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _read_pickle(path: str) -> np.ndarray:
    with open(path, "rb") as file:
        return np.asarray(pickle.load(file, encoding="latin1"), dtype=np.float64)


def _read_fold_file(path: str) -> list[list[int]]:
    with open(path, "r", encoding="utf-8") as file:
        parsed = ast.literal_eval(file.read().strip())
    if parsed and all(isinstance(item, int) for item in parsed):
        return [parsed]
    return parsed


def get_split_indices(
    dataset_name: str,
    fold_index: int = 0,
    use_dataset_folds: bool = True,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 42,
) -> tuple[dict[str, list[int]], str]:
    dataset_dir = os.path.join(MODULE_ROOT, "data", dataset_name.lower().strip())
    y = _read_pickle(os.path.join(dataset_dir, "Y"))
    dataset_size = int(y.size if y.ndim == 1 else y.shape[0] * y.shape[1])
    folds_dir = os.path.join(dataset_dir, "folds")
    fold_paths = {
        "train": os.path.join(folds_dir, "train_fold_setting1.txt"),
        "valid": os.path.join(folds_dir, "valid_fold_setting1.txt"),
        "test": os.path.join(folds_dir, "test_fold_setting1.txt"),
    }
    if use_dataset_folds and all(os.path.exists(path) for path in fold_paths.values()):
        folds = {name: _read_fold_file(path) for name, path in fold_paths.items()}
        num_folds = min(len(value) for value in folds.values())
        if fold_index < 0 or fold_index >= num_folds:
            raise ValueError(f"Invalid fold_index={fold_index}. Available folds: 0..{num_folds - 1}.")
        return {name: values[fold_index] for name, values in folds.items()}, "dataset_folds"

    indices = list(range(dataset_size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    test_size = max(1, int(math.floor(test_split * dataset_size)))
    val_size = max(1, int(math.floor(val_split * dataset_size))) if val_split > 0 else 1
    if test_size + val_size >= dataset_size:
        raise ValueError("Dataset too small for requested random split.")
    return {
        "test": indices[:test_size],
        "valid": indices[test_size:test_size + val_size],
        "train": indices[test_size + val_size:],
    }, "random"


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "min": float(np.min(values)) if values.size else float("nan"),
        "max": float(np.max(values)) if values.size else float("nan"),
        "mean": float(np.mean(values)) if values.size else float("nan"),
        "std": float(np.std(values)) if values.size else float("nan"),
    }


def audit_dataset_splits(
    dataset_name: str = "mpro_urv",
    fold_index: int = 0,
    use_dataset_folds: bool = True,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 42,
    output_dir: str | None = None,
    model_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_name = dataset_name.lower().strip()
    dataset_dir = os.path.join(MODULE_ROOT, "data", dataset_name)
    ligand_path = os.path.join(dataset_dir, "ligands_can.txt")
    motif_path = os.path.join(dataset_dir, "motif2.txt")
    y_path = os.path.join(dataset_dir, "Y")
    metadata_path = os.path.join(dataset_dir, "metadata.json")

    for path in (ligand_path, motif_path, y_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required WideDTA dataset file not found: {path}")

    ligands = _read_json(ligand_path)
    ligand_items = list(ligands.items())
    y = _read_pickle(y_path).reshape(-1)
    dataset_size = int(y.shape[0])
    splits, split_mode = get_split_indices(dataset_name, fold_index, use_dataset_folds, val_split, test_split, seed)

    for name, indices in splits.items():
        if not indices:
            raise ValueError(f"{name} split is empty.")
        invalid = [index for index in indices if index < 0 or index >= dataset_size]
        if invalid:
            raise ValueError(f"{name} split has indices outside dataset range: {invalid[:10]}")

    overlaps = {
        "train_valid": sorted(set(splits["train"]) & set(splits["valid"])),
        "train_test": sorted(set(splits["train"]) & set(splits["test"])),
        "valid_test": sorted(set(splits["valid"]) & set(splits["test"])),
    }
    if any(overlaps.values()):
        raise ValueError(
            "Dataset split leakage detected: "
            f"train/valid={len(overlaps['train_valid'])}, "
            f"train/test={len(overlaps['train_test'])}, valid/test={len(overlaps['valid_test'])}"
        )

    def ligand_for(index: int) -> tuple[str | None, str | None]:
        if index < len(ligand_items):
            key, smiles = ligand_items[index]
            return str(key), str(smiles)
        return None, None

    duplicate_smiles = {}
    duplicate_ids = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        left_ids, left_smiles = zip(*(ligand_for(index) for index in splits[left]))
        right_ids, right_smiles = zip(*(ligand_for(index) for index in splits[right]))
        duplicate_ids[f"{left}_{right}"] = sorted(set(left_ids) & set(right_ids) - {None})
        duplicate_smiles[f"{left}_{right}"] = sorted(set(left_smiles) & set(right_smiles) - {None})

    train_mean = float(np.mean(y[splits["train"]]))
    baseline = {}
    for name in ("valid", "test"):
        true = y[splits[name]]
        pred = np.full_like(true, train_mean, dtype=np.float64)
        baseline[name] = {
            "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
            "mae": float(np.mean(np.abs(pred - true))),
        }

    metadata = _read_json(metadata_path) if os.path.exists(metadata_path) else {}
    warnings = ["Test set not used for model selection."]
    if metadata.get("motif_source") == "protein_sequence" or metadata.get("motif_mode") == "technical_motif_baseline":
        warnings.append("Technical motif baseline: motif2.txt duplicates protein sequence and is not biological motif extraction.")
    if any(duplicate_smiles.values()):
        warnings.append("Duplicate ligand strings/SMILES occur across splits; inspect whether this is acceptable for ligand-level generalization.")
    if model_metrics:
        for split_name in ("valid", "test"):
            key = f"{split_name}_rmse" if split_name == "test" else "val_rmse"
            rmse = model_metrics.get(key)
            base = baseline[split_name]["rmse"]
            if rmse is not None and math.isfinite(float(rmse)) and float(rmse) < 0.5 * base:
                warnings.append(f"{split_name} RMSE is dramatically better than naive train-mean baseline; manual inspection required.")

    audit = {
        "model": "WideDTA",
        "dataset": dataset_name,
        "dataset_dir": dataset_dir,
        "split_mode": split_mode,
        "fold_index": fold_index if split_mode == "dataset_folds" else None,
        "dataset_size": dataset_size,
        "motif_mode": metadata.get("motif_mode", metadata.get("motif_source")),
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "overlap_counts": {name: len(indices) for name, indices in overlaps.items()},
        "duplicate_sample_ids": {name: values[:20] for name, values in duplicate_ids.items()},
        "duplicate_ligand_strings": {name: values[:20] for name, values in duplicate_smiles.items()},
        "label_distribution": {name: _distribution(y[indices]) for name, indices in splits.items()},
        "naive_train_mean_baseline": baseline,
        "warnings": warnings,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audit_path = os.path.join(output_dir, f"split_audit_widedta_{timestamp}.json")
        with open(audit_path, "w", encoding="utf-8") as file:
            json.dump(audit, file, indent=4, default=str)
        audit["audit_path"] = audit_path

    return audit
