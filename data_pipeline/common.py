import ast
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch


def load_split_txt(path):
    with open(path, "r") as f:
        return ast.literal_eval(f.read())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_all_trials_results_csv(results, save_directory):
    output_dir = Path(save_directory) / "hyperparameter_tuning_all_trials"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "all_trials_results.csv"

    rows = []

    for trial_index, result in enumerate(results):
        row = {
            "trial_index": trial_index,
            "trial_path": str(result.path),
        }

        for key, value in result.config.items():
            row[f"param_{key}"] = value

        metrics = result.metrics

        row["mean_val_rmse"] = metrics.get("mean_val_rmse")
        row["std_val_rmse"] = metrics.get("std_val_rmse")

        row["mean_test_rmse"] = metrics.get("mean_test_rmse")
        row["std_test_rmse"] = metrics.get("std_test_rmse")

        row["mean_test_pearson"] = metrics.get("mean_test_pearson")
        row["std_test_pearson"] = metrics.get("std_test_pearson")
        row["mean_train_rmse_norm"] = metrics.get("mean_train_rmse_norm")
        row["std_train_rmse_norm"] = metrics.get("std_train_rmse_norm")

        rows.append(row)

    if not rows:
        return csv_path

    fieldnames = sorted(rows[0].keys())

    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def save_csv_for_planet(results,save_directory):
    output_dir = Path(save_directory) / "hyperparameter_tuning_all_trials"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "all_trials_results.csv"

    rows = []

    for trial_index, result in enumerate(results):
        metrics = result.metrics or {}

        row = {
            "trial_index": trial_index,
            "trial_path": str(result.path),
            "train_rmse": metrics.get("train_rmse"),
            "train_pearson": metrics.get("train_pearson"),
            "val_rmse": metrics.get("val_rmse"),
            "val_pearson": metrics.get("val_pearson"),
            "test_rmse": metrics.get("test_rmse"),
            "test_pearson": metrics.get("test_pearson"),
            "best_epoch": metrics.get("best_epoch"),
            "best_val_rmse": metrics.get("best_val_rmse"),
            "training_time": metrics.get("training_time"),
            "skipped_train_eval": metrics.get("skipped_train_eval"),
            "skipped_val_eval": metrics.get("skipped_val_eval"),
            "skipped_test_eval": metrics.get("skipped_test_eval"),
        }

        for key, value in result.config.items():
            row[f"param_{key}"] = value

        rows.append(row)
    if not rows:
        return csv_path

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def safe_computation_pearson(pred, label):
    pred = np.asarray(pred).reshape(-1)
    label = np.asarray(label).reshape(-1)

    if len(pred) < 2:
        return 0.0
    if np.std(pred) == 0 or np.std(label) == 0:
        return 0.0
    return np.corrcoef(pred, label)[0, 1]


def ensure_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, path):
    path = Path(path)
    ensure_directory(path.parent)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path


def get_input_files(pdb_dir, sdf_dir, pdb_id):
    pdb_dir = Path(pdb_dir)
    sdf_dir = Path(sdf_dir)

    pdb_id = str(pdb_id).strip().upper()
    pdb_id_lower = pdb_id.lower()

    protein_candidates = [
        pdb_dir / f"{pdb_id}.pdb",
        pdb_dir / f"{pdb_id}_protein.pdb",
        pdb_dir / f"{pdb_id_lower}.pdb",
        pdb_dir / f"{pdb_id_lower}_protein.pdb",
    ]

    ligand_candidates = [
        sdf_dir / f"{pdb_id}.sdf",
        sdf_dir / f"{pdb_id}_ligand.sdf",
        sdf_dir / f"{pdb_id_lower}.sdf",
        sdf_dir / f"{pdb_id_lower}_ligand.sdf",
    ]

    protein_path = next((path for path in protein_candidates if path.is_file()), None)
    ligand_path = next((path for path in ligand_candidates if path.is_file()), None)

    if protein_path is None:
        raise FileNotFoundError(f"PDB file not found for {pdb_id}. Tried: {protein_candidates}")

    if ligand_path is None:
        raise FileNotFoundError(f"SDF file not found for {pdb_id}. Tried: {ligand_candidates}")

    return protein_path, ligand_path


def copy_file(src, dst, overwrite=False):
    src = Path(src)
    dst = Path(dst)

    ensure_directory(dst.parent)

    if dst.exists():
        if not overwrite:
            return dst
        dst.unlink()

    shutil.copy2(src, dst)
    return dst


def write_split_csv(path, ids, pic50, split_type="REFINED"):
    path = Path(path)
    ensure_directory(path.parent)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["", "PDB_code", "pK", "type"],
        )
        writer.writeheader()

        for index, pdb_id in enumerate(ids):
            writer.writerow(
                {
                    "": index,
                    "PDB_code": pdb_id,
                    "pK": pic50[pdb_id],
                    "type": split_type,
                }
            )

    return path
