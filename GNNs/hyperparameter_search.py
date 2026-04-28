"""
@file hyperparameter_search.py
@brief Generic hyperparameter search for the GNN models.

This module runs hyperparameter search over the models defined in
GNNs/model_trainer.py:

- GIN
- GINE
- GAT
- EGAT
- GraphTransformer

It is designed to work without modifying the professor's training code.
Each trial is executed inside its own run directory, so functions that save
to relative folders such as "Modelos" and "Resultados" remain isolated per trial.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import random
import shutil
import time
import traceback
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch

from GNNs.model_trainer import train_and_save_model
from GNNs.model_tester import test_model_on_directory


SUPPORTED_MODELS = [
    "GIN",
    "GINE",
    "GAT",
    "EGAT",
    "GraphTransformer",
]

DEFAULT_SEARCH_SPACE = {
    "lr": [1e-3, 5e-4, 1e-4],
    "batch_size": [16, 32],
    "hidden_dim": [64, 128],
    "num_layers": [2, 3],
    "atom_emb_dim": [0.4],
    "hibrid_emb_dim": [0.5],
    "bond_emb_dim": [1],
}


@contextmanager
def working_directory(path: str | Path):
    """
    Temporarily change the current working directory.

    This is necessary because the professor's training and testing functions
    save files using relative paths.
    """
    previous_dir = os.getcwd()
    os.makedirs(path, exist_ok=True)

    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous_dir)


def seed_everything(seed: int) -> None:
    """
    Set common random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> None:
    """
    Create a directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def validate_input_paths(
    train_sdf_dir: str,
    target_file: str,
    eval_sdf_dir: str,
    eval_targets_file: str,
) -> None:
    """
    Validate all input paths before launching expensive training loops.
    """
    checks = [
        ("train_sdf_dir", train_sdf_dir, "dir"),
        ("target_file", target_file, "file"),
        ("eval_sdf_dir", eval_sdf_dir, "dir"),
        ("eval_targets_file", eval_targets_file, "file"),
    ]

    errors = []

    for name, path, expected_type in checks:
        p = Path(path)

        if expected_type == "dir" and not p.is_dir():
            errors.append(f"{name} is not a directory or does not exist: {path}")

        if expected_type == "file" and not p.is_file():
            errors.append(f"{name} is not a file or does not exist: {path}")

    if errors:
        raise FileNotFoundError("\n".join(errors))


def build_trial_uid(payload: Dict[str, Any]) -> str:
    """
    Build a stable short hash for one trial configuration.
    """
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def generate_grid(search_space: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    """
    Generate all parameter combinations from a search space dictionary.
    """
    keys = list(search_space.keys())
    values = [search_space[key] for key in keys]

    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def read_previous_trials(csv_path: str | Path) -> Dict[str, str]:
    """
    Read already executed trials.

    Returns:
        dict: trial_uid -> last known status
    """
    if not os.path.exists(csv_path):
        return {}

    status_by_uid = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            trial_uid = row.get("trial_uid")
            status = row.get("status")

            if trial_uid:
                status_by_uid[trial_uid] = status or ""

    return status_by_uid


def append_csv_row(csv_path: str | Path, row: Dict[str, Any], fieldnames: list[str]) -> None:
    """
    Append one row to a CSV file, creating it with headers if needed.
    """
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")

        if not file_exists:
            writer.writeheader()

        clean_row = {}

        for key in fieldnames:
            value = row.get(key, "")

            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=False)

            clean_row[key] = value

        writer.writerow(clean_row)


def find_latest_checkpoint(run_dir: str | Path) -> Optional[str]:
    """
    Find the most recent checkpoint file generated inside one run directory.
    """
    run_path = Path(run_dir)

    candidates = []

    for extension in ("*.pt", "*.pth", "*.ckpt"):
        candidates.extend(run_path.rglob(extension))

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0].resolve())


def maybe_path_from_return(value: Any) -> Optional[str]:
    """
    Try to recover a model path from the return value of train_and_save_model.
    """
    if isinstance(value, str) and os.path.exists(value):
        return str(Path(value).resolve())

    if isinstance(value, Path) and value.exists():
        return str(value.resolve())

    if isinstance(value, dict):
        for key in ("model_path", "checkpoint_path", "path", "saved_model"):
            path = value.get(key)

            if isinstance(path, str) and os.path.exists(path):
                return str(Path(path).resolve())

    return None


def normalize_text(value: str) -> str:
    """
    Normalize text for robust matching.
    """
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))

    for char in [" ", "_", "-", ".", "(", ")", "[", "]", "{", "}", "/", "\\", ":"]:
        value = value.replace(char, "")

    return value


def normalize_metric_name(name: str) -> str:
    """
    Normalize metric names to a stable format.
    """
    clean = normalize_text(name)

    aliases = {
        "rmse": "RMSE",
        "rootmeansquarederror": "RMSE",
        "mse": "MSE",
        "meansquarederror": "MSE",
        "mae": "MAE",
        "meanabsoluteerror": "MAE",
        "pearson": "Pearson",
        "pearsonr": "Pearson",
        "pearsoncoefficient": "Pearson",
        "rp": "Pearson",
        "spearman": "Spearman",
        "spearmanr": "Spearman",
        "r2": "R2",
        "r2score": "R2",
        "r2coefficient": "R2",
    }

    return aliases.get(clean, name)


def metrics_from_dict(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract metrics from a dictionary returned by a tester function.
    """
    metrics = {}

    for key, value in data.items():
        metric_name = normalize_metric_name(str(key))

        if metric_name in {"RMSE", "MSE", "MAE", "Pearson", "Spearman", "R2"}:
            try:
                metrics[metric_name] = float(value)
            except Exception:
                pass

    return metrics


def read_csv_flexible(csv_path: str | Path) -> pd.DataFrame:
    """
    Read a CSV file with basic delimiter fallback.
    """
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.read_csv(csv_path, sep=None, engine="python")


def find_prediction_csv(run_dir: str | Path) -> Optional[str]:
    """
    Find the CSV file that probably contains predictions.
    """
    run_path = Path(run_dir)

    csv_files = list(run_path.rglob("*.csv"))

    if not csv_files:
        return None

    preferred_keywords = [
        "predicciones",
        "prediccion",
        "prediction",
        "predictions",
        "resultado",
        "resultados",
        "results",
        "metrics",
        "metricas",
        "resumen",
    ]

    scored = []

    for path in csv_files:
        name = normalize_text(path.name)
        score = sum(keyword in name for keyword in preferred_keywords)

        # Penalize files that are clearly not prediction outputs.
        if "trialconfig" in name:
            score -= 10

        scored.append((score, path.stat().st_mtime, path))

    scored.sort(reverse=True)
    return str(scored[0][2].resolve())


def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Compute regression metrics from true and predicted values.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if len(y_true) != len(y_pred):
        return {}

    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return {}

    diff = y_true - y_pred

    metrics = {
        "MSE": float(np.mean(diff ** 2)),
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "MAE": float(np.mean(np.abs(diff))),
    }

    if len(y_true) > 1:
        metrics["Pearson"] = float(np.corrcoef(y_true, y_pred)[0, 1])
        metrics["Spearman"] = float(
            pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
        )

        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        metrics["R2"] = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

    return metrics


def metrics_from_metrics_csv(csv_path: str | Path) -> Dict[str, float]:
    """
    Try to extract metrics from a CSV that directly contains metric values.

    Supports formats like:
    Metric,Value
    RMSE,0.98
    Pearson,0.67
    """
    try:
        df = read_csv_flexible(csv_path)
    except Exception:
        return {}

    if df.empty:
        return {}

    normalized_cols = {normalize_text(col): col for col in df.columns}

    metric_col = None
    value_col = None

    for key, original in normalized_cols.items():
        if key in {"metric", "metrica", "name", "nombre"}:
            metric_col = original
        if key in {"value", "valor", "mean", "media"}:
            value_col = original

    if metric_col is None or value_col is None:
        return {}

    metrics = {}

    for _, row in df.iterrows():
        metric_name = normalize_metric_name(str(row[metric_col]))

        if metric_name in {"RMSE", "MSE", "MAE", "Pearson", "Spearman", "R2"}:
            try:
                metrics[metric_name] = float(row[value_col])
            except Exception:
                pass

    return metrics


def metrics_from_prediction_csv(csv_path: str | Path) -> Dict[str, float]:
    """
    Compute metrics from a CSV containing real and predicted values.

    Robust against Spanish and English column names.
    """
    try:
        df = read_csv_flexible(csv_path)
    except Exception:
        return {}

    if df.empty:
        return {}

    # First try: maybe the CSV already stores metrics.
    direct_metrics = metrics_from_metrics_csv(csv_path)
    if direct_metrics:
        return direct_metrics

    normalized_columns = {
        normalize_text(col): col
        for col in df.columns
    }

    real_candidates = [
        "yreal",
        "ytrue",
        "true",
        "real",
        "target",
        "targetvalue",
        "valorreal",
        "valorverdadero",
        "valorexperimental",
        "experimental",
        "pic50",
        "pic50real",
        "pIC50real",
    ]

    pred_candidates = [
        "ypred",
        "pred",
        "preds",
        "prediction",
        "predictions",
        "predicted",
        "valorpredicho",
        "prediccion",
        "predicciones",
        "valorpredecido",
        "estimado",
        "valorestimado",
        "pic50predicho",
        "pIC50predicho",
    ]

    real_col = None
    pred_col = None

    for candidate in real_candidates:
        key = normalize_text(candidate)

        if key in normalized_columns:
            real_col = normalized_columns[key]
            break

    for candidate in pred_candidates:
        key = normalize_text(candidate)

        if key in normalized_columns:
            pred_col = normalized_columns[key]
            break

    # Fallback: substring matching.
    if real_col is None:
        for norm_col, original_col in normalized_columns.items():
            if (
                "real" in norm_col
                or "true" in norm_col
                or "target" in norm_col
                or "experimental" in norm_col
            ):
                real_col = original_col
                break

    if pred_col is None:
        for norm_col, original_col in normalized_columns.items():
            if (
                "pred" in norm_col
                or "prediction" in norm_col
                or "prediccion" in norm_col
                or "estimado" in norm_col
            ):
                pred_col = original_col
                break

    # Last fallback: use last two numeric columns.
    if real_col is None or pred_col is None:
        numeric_columns = []

        for col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().sum() > 0:
                numeric_columns.append(col)

        if len(numeric_columns) >= 2:
            real_col = numeric_columns[-2]
            pred_col = numeric_columns[-1]

    if real_col is None or pred_col is None:
        print(f"[WARNING] Could not identify real/prediction columns in: {csv_path}")
        print(f"[WARNING] Available columns: {list(df.columns)}")
        return {}

    y_true = pd.to_numeric(df[real_col], errors="coerce")
    y_pred = pd.to_numeric(df[pred_col], errors="coerce")

    valid_mask = y_true.notna() & y_pred.notna()
    y_true = y_true[valid_mask].to_numpy(dtype=float)
    y_pred = y_pred[valid_mask].to_numpy(dtype=float)

    if len(y_true) == 0:
        print(f"[WARNING] No valid numeric rows found in: {csv_path}")
        print(f"[WARNING] Real column: {real_col}")
        print(f"[WARNING] Pred column: {pred_col}")
        return {}

    return compute_regression_metrics(y_true, y_pred)


def extract_metrics(test_return: Any, run_dir: str | Path) -> Dict[str, float]:
    """
    Extract metrics from the tester return value or generated CSV files.
    """

    # Case 1: tester returns a dictionary.
    if isinstance(test_return, dict):
        metrics = metrics_from_dict(test_return)

        if metrics:
            return metrics

        real_keys = ["y_real", "y_true", "true", "target", "labels", "real"]
        pred_keys = ["y_pred", "pred", "preds", "prediction", "predictions", "predicted"]

        y_true = None
        y_pred = None

        for key in real_keys:
            if key in test_return:
                y_true = test_return[key]
                break

        for key in pred_keys:
            if key in test_return:
                y_pred = test_return[key]
                break

        if y_true is not None and y_pred is not None:
            metrics = compute_regression_metrics(y_true, y_pred)

            if metrics:
                return metrics

    # Case 2: tester returns a pandas DataFrame.
    if isinstance(test_return, pd.DataFrame):
        temp_csv = Path(run_dir) / "_tester_return_dataframe.csv"
        test_return.to_csv(temp_csv, index=False)

        metrics = metrics_from_prediction_csv(temp_csv)

        if metrics:
            return metrics

    # Case 3: tester returns a CSV path.
    if isinstance(test_return, (str, Path)) and os.path.exists(test_return):
        path = Path(test_return)

        if path.suffix.lower() == ".csv":
            metrics = metrics_from_prediction_csv(path)

            if metrics:
                return metrics

    # Case 4: tester returns tuple/list.
    if isinstance(test_return, (tuple, list)):
        numeric_arrays = []

        for item in test_return:
            try:
                arr = np.asarray(item, dtype=float).reshape(-1)

                if len(arr) > 1:
                    numeric_arrays.append(arr)
            except Exception:
                pass

        if len(numeric_arrays) >= 2:
            y_true = numeric_arrays[-2]
            y_pred = numeric_arrays[-1]

            metrics = compute_regression_metrics(y_true, y_pred)

            if metrics:
                return metrics

        scalar_values = []

        for item in test_return:
            try:
                scalar_values.append(float(item))
            except Exception:
                pass

        if len(scalar_values) == 3:
            return {
                "RMSE": scalar_values[0],
                "Pearson": scalar_values[1],
                "Spearman": scalar_values[2],
            }

        if len(scalar_values) == 4:
            return {
                "RMSE": scalar_values[0],
                "MAE": scalar_values[1],
                "Pearson": scalar_values[2],
                "Spearman": scalar_values[3],
            }

    # Case 5: tester generated a CSV inside the run directory.
    csv_path = find_prediction_csv(run_dir)

    if csv_path:
        metrics = metrics_from_prediction_csv(csv_path)

        if metrics:
            return metrics

        print(f"[WARNING] CSV found but metrics could not be extracted: {csv_path}")

    return {}


def is_valid_metric(value: Any) -> bool:
    """
    Check if a metric value can be used for ranking.
    """
    try:
        value = float(value)
    except Exception:
        return False

    return not np.isnan(value) and not np.isinf(value)


def is_better(
    candidate: Dict[str, float],
    current_best: Optional[Dict[str, float]],
    objective_metric: str,
    objective_mode: str,
) -> bool:
    """
    Decide if one candidate is better than the current best.
    """
    if objective_metric not in candidate:
        return False

    candidate_value = candidate[objective_metric]

    if not is_valid_metric(candidate_value):
        return False

    if current_best is None:
        return True

    best_value = current_best.get(objective_metric)

    if not is_valid_metric(best_value):
        return True

    if objective_mode == "min":
        return candidate_value < best_value

    if objective_mode == "max":
        return candidate_value > best_value

    raise ValueError("objective_mode must be either 'min' or 'max'.")


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """
    Save a dictionary as JSON.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)


def save_yaml_if_available(data: Dict[str, Any], path: str | Path) -> None:
    """
    Save a dictionary as YAML if PyYAML is installed.
    """
    try:
        import yaml
    except Exception:
        return

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def run_hyperparameter_search(
    train_sdf_dir: str,
    target_file: str,
    output_root: str,
    model_names: Optional[list[str]] = None,
    search_space: Optional[Dict[str, list[Any]]] = None,
    eval_sdf_dir: Optional[str] = None,
    eval_targets_file: Optional[str] = None,
    epochs: int = 20,
    patience: int = 0,
    valid_split: float = 0.2,
    objective_metric: str = "RMSE",
    objective_mode: str = "min",
    resume: bool = True,
    rerun_failed: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run hyperparameter search for several GNN models.
    """
    if model_names is None:
        model_names = SUPPORTED_MODELS

    invalid_models = [name for name in model_names if name not in SUPPORTED_MODELS]

    if invalid_models:
        raise ValueError(f"Unsupported model names: {invalid_models}")

    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE

    objective_metric = normalize_metric_name(objective_metric)

    if objective_mode not in {"min", "max"}:
        raise ValueError("objective_mode must be either 'min' or 'max'.")

    train_sdf_dir = str(Path(train_sdf_dir).resolve())
    target_file = str(Path(target_file).resolve())

    if eval_sdf_dir is None:
        eval_sdf_dir = train_sdf_dir
    else:
        eval_sdf_dir = str(Path(eval_sdf_dir).resolve())

    if eval_targets_file is None:
        eval_targets_file = target_file
    else:
        eval_targets_file = str(Path(eval_targets_file).resolve())

    validate_input_paths(
        train_sdf_dir=train_sdf_dir,
        target_file=target_file,
        eval_sdf_dir=eval_sdf_dir,
        eval_targets_file=eval_targets_file,
    )

    output_root = Path(output_root).resolve()
    runs_dir = output_root / "runs"

    ensure_dir(output_root)
    ensure_dir(runs_dir)

    trials_csv = output_root / "trials.csv"
    failed_trials_csv = output_root / "failed_trials.csv"
    best_config_json = output_root / "best_config.json"
    best_config_yaml = output_root / "best_config.yaml"

    param_keys = list(search_space.keys())

    csv_fields = [
        "trial_uid",
        "trial_index",
        "trial_name",
        "model_type",
        "status",
        "error_message",
        "checkpoint_path",
        "elapsed_seconds",
        "objective_metric",
        "objective_mode",
        "RMSE",
        "MSE",
        "MAE",
        "Pearson",
        "Spearman",
        "R2",
        "epochs",
        "patience",
        "valid_split",
        "seed",
    ] + param_keys

    previous_trials = read_previous_trials(trials_csv)

    best_metrics = None
    best_payload = None

    all_trial_configs = []

    for model_type in model_names:
        for params in generate_grid(search_space):
            trial_config = {
                "model_type": model_type,
                "train_sdf_dir": train_sdf_dir,
                "target_file": target_file,
                "eval_sdf_dir": eval_sdf_dir,
                "eval_targets_file": eval_targets_file,
                "epochs": epochs,
                "patience": patience,
                "valid_split": valid_split,
                "seed": seed,
                **params,
            }

            trial_uid = build_trial_uid(trial_config)

            all_trial_configs.append(
                {
                    "trial_uid": trial_uid,
                    "config": trial_config,
                }
            )

    total_trials = len(all_trial_configs)

    for trial_index, item in enumerate(all_trial_configs, start=1):
        trial_uid = item["trial_uid"]
        config = item["config"]

        model_type = config["model_type"]
        params = {key: config[key] for key in param_keys}

        previous_status = previous_trials.get(trial_uid)

        if resume and previous_status:
            if previous_status == "success":
                print(f"[SKIP] Trial {trial_index}/{total_trials} already completed: {trial_uid}")
                continue

            if previous_status != "success" and not rerun_failed:
                print(f"[SKIP] Trial {trial_index}/{total_trials} already failed: {trial_uid}")
                continue

        trial_name = f"trial_{trial_index:04d}_{model_type}"
        run_dir = runs_dir / trial_name

        if run_dir.exists() and (not resume or rerun_failed):
            shutil.rmtree(run_dir)

        ensure_dir(run_dir)

        save_json(config, run_dir / "trial_config.json")

        print("\n" + "=" * 70)
        print(f"Trial {trial_index}/{total_trials}: {trial_name}")
        print(f"UID: {trial_uid}")
        print(f"Model: {model_type}")
        print(f"Params: {params}")
        print("=" * 70)

        start_time = time.time()
        checkpoint_path = ""
        metrics = {}

        seed_everything(seed + trial_index)

        try:
            with working_directory(run_dir):
                train_return = train_and_save_model(
                    sdf_dir=train_sdf_dir,
                    target_file=target_file,
                    model_type=model_type,
                    epochs=epochs,
                    model_name=trial_name,
                    batch_size=int(params.get("batch_size", 32)),
                    lr=float(params.get("lr", 0.001)),
                    valid_split=float(config.get("valid_split", valid_split)),
                    hidden_dim=int(params.get("hidden_dim", 64)),
                    num_layers=int(params.get("num_layers", 3)),
                    patience=int(config.get("patience", patience)),
                    atom_emb_dim=float(params.get("atom_emb_dim", 0.4)),
                    hibrid_emb_dim=float(params.get("hibrid_emb_dim", 0.5)),
                    bond_emb_dim=float(params.get("bond_emb_dim", 1)),
                )

            checkpoint_path = maybe_path_from_return(train_return) or find_latest_checkpoint(run_dir)

            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"No checkpoint file was found inside trial directory: {run_dir}"
                )

            with working_directory(run_dir):
                test_return = test_model_on_directory(
                    checkpoint_path=checkpoint_path,
                    sdf_dir=eval_sdf_dir,
                    targets_file=eval_targets_file,
                )

            metrics = extract_metrics(test_return, run_dir)

            if objective_metric not in metrics:
                csv_candidate = find_prediction_csv(run_dir)

                raise RuntimeError(
                    f"Objective metric '{objective_metric}' could not be extracted. "
                    f"Extracted metrics: {metrics}. "
                    f"CSV candidate: {csv_candidate}"
                )

            elapsed = time.time() - start_time

            row = {
                "trial_uid": trial_uid,
                "trial_index": trial_index,
                "trial_name": trial_name,
                "model_type": model_type,
                "status": "success",
                "error_message": "",
                "checkpoint_path": checkpoint_path,
                "elapsed_seconds": round(elapsed, 3),
                "objective_metric": objective_metric,
                "objective_mode": objective_mode,
                "epochs": epochs,
                "patience": patience,
                "valid_split": valid_split,
                "seed": seed,
                **params,
                **metrics,
            }

            append_csv_row(trials_csv, row, csv_fields)

            if is_better(metrics, best_metrics, objective_metric, objective_mode):
                best_metrics = metrics

                best_payload = {
                    "status": "success",
                    "trial_uid": trial_uid,
                    "trial_index": trial_index,
                    "trial_name": trial_name,
                    "model_type": model_type,
                    "checkpoint_path": checkpoint_path,
                    "objective_metric": objective_metric,
                    "objective_mode": objective_mode,
                    "metrics": metrics,
                    "hyperparameters": params,
                    "epochs": epochs,
                    "patience": patience,
                    "valid_split": valid_split,
                    "seed": seed,
                    "run_dir": str(run_dir),
                }

                save_json(best_payload, best_config_json)
                save_yaml_if_available(best_payload, best_config_yaml)

            print(f"[OK] {trial_name}")
            print(f"Checkpoint: {checkpoint_path}")
            print(f"Metrics: {metrics}")

        except Exception as exc:
            elapsed = time.time() - start_time
            error_message = traceback.format_exc()

            row = {
                "trial_uid": trial_uid,
                "trial_index": trial_index,
                "trial_name": trial_name,
                "model_type": model_type,
                "status": "failed",
                "error_message": error_message[:3000],
                "checkpoint_path": checkpoint_path,
                "elapsed_seconds": round(elapsed, 3),
                "objective_metric": objective_metric,
                "objective_mode": objective_mode,
                "epochs": epochs,
                "patience": patience,
                "valid_split": valid_split,
                "seed": seed,
                **params,
                **metrics,
            }

            append_csv_row(trials_csv, row, csv_fields)
            append_csv_row(failed_trials_csv, row, csv_fields)

            print(f"[FAILED] {trial_name}")
            print(str(exc))

    if best_payload is None:
        return {
            "status": "failed",
            "message": "No valid trial completed successfully.",
            "trials_csv": str(trials_csv),
            "failed_trials_csv": str(failed_trials_csv),
            "best_config_json": str(best_config_json),
            "best_config_yaml": str(best_config_yaml),
        }

    return {
        "status": "success",
        "message": "Hyperparameter search completed.",
        "best_trial": best_payload,
        "trials_csv": str(trials_csv),
        "failed_trials_csv": str(failed_trials_csv),
        "best_config_json": str(best_config_json),
        "best_config_yaml": str(best_config_yaml),
    }

def print_trials_summary_table(
    trials_csv: str | Path,
    top_n: int = 20,
    sort_by: str = "RMSE",
    ascending: bool = True,
) -> None:
    """
    Print a clean terminal summary table from trials.csv.
    """
    trials_csv = Path(trials_csv)

    if not trials_csv.exists():
        print(f"No trials CSV found: {trials_csv}")
        return

    df = pd.read_csv(trials_csv)

    if df.empty:
        print("trials.csv is empty.")
        return

    success_df = df[df["status"] == "success"].copy()

    if success_df.empty:
        print("No successful trials found.")
        return

    numeric_cols = [
        "RMSE",
        "MSE",
        "MAE",
        "Pearson",
        "Spearman",
        "R2",
        "lr",
        "batch_size",
        "hidden_dim",
        "num_layers",
        "epochs",
        "patience",
    ]

    for col in numeric_cols:
        if col in success_df.columns:
            success_df[col] = pd.to_numeric(success_df[col], errors="coerce")

    if sort_by in success_df.columns:
        success_df = success_df.sort_values(sort_by, ascending=ascending)

    display_cols = [
        "trial_index",
        "trial_name",
        "model_type",
        "RMSE",
        "MAE",
        "Pearson",
        "Spearman",
        "R2",
        "lr",
        "batch_size",
        "hidden_dim",
        "num_layers",
    ]

    display_cols = [col for col in display_cols if col in success_df.columns]

    table = success_df[display_cols].head(top_n)

    print("\n" + "=" * 120)
    print(f"TOP {min(top_n, len(table))} TRIALS SORTED BY {sort_by}")
    print("=" * 120)
    print(table.to_string(index=False))
    print("=" * 120)

    best = success_df.iloc[0]

    print("\nBEST CONFIGURATION")
    print("-" * 120)
    print(f"Trial      : {best.get('trial_name')}")
    print(f"Model      : {best.get('model_type')}")
    print(f"RMSE       : {best.get('RMSE'):.6f}")
    print(f"MAE        : {best.get('MAE'):.6f}")
    print(f"Pearson    : {best.get('Pearson'):.6f}")
    print(f"Spearman   : {best.get('Spearman'):.6f}")
    print(f"R2         : {best.get('R2'):.6f}")
    print(f"LR         : {best.get('lr')}")
    print(f"Batch size : {best.get('batch_size')}")
    print(f"Hidden dim : {best.get('hidden_dim')}")
    print(f"Num layers : {best.get('num_layers')}")
    print(f"Checkpoint : {best.get('checkpoint_path')}")
    print("-" * 120)

if __name__ == "__main__":
    results = run_hyperparameter_search(
        train_sdf_dir="/home/mohamed/Studies/stage/MPro-URV_Version2/MPro-URV_Version2/Ligand/Ligand_SDF",
        target_file="/home/mohamed/Studies/stage/MPro-URV_Version2/MPro-URV_Version2/pIC50.txt",
        eval_sdf_dir="/home/mohamed/Studies/stage/MPro-URV_Version2/MPro-URV_Version2/Ligand/Ligand_SDF",
        eval_targets_file="/home/mohamed/Studies/stage/MPro-URV_Version2/MPro-URV_Version2/pIC50.txt",
        output_root="hyperparameter_Search",

        model_names=[
            "GIN",
            "GINE",
            "GAT",
            "EGAT",
            "GraphTransformer",
        ],

        search_space={
            "lr": [1e-3, 5e-4, 1e-4],
            "batch_size": [16, 32],
            "hidden_dim": [64, 128],
            "num_layers": [2, 3],
            "atom_emb_dim": [0.4],
            "hibrid_emb_dim": [0.5],
            "bond_emb_dim": [1],
        },

        epochs=50,
        patience=10,
        valid_split=0.2,

        objective_metric="RMSE",
        objective_mode="min",

        resume=True,
        rerun_failed=True,
        seed=42,
    )

    print("\nFINAL STATUS")
    print(results["status"])
    print(results["message"])
    print(f"Trials CSV: {results['trials_csv']}")

    print_trials_summary_table(
        trials_csv=results["trials_csv"],
        top_n=20,
        sort_by="RMSE",
        ascending=True,
    )
