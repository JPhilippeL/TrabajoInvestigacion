"""
@file widedta_hyperparameter_search.py
@author Mohamed EL BOUKHIARI
@brief Hyperparameter search pipeline for the WideDTA module.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import shutil
import time
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from WideDTA.Core.widedta_audit import audit_dataset_splits
from WideDTA.Core.widedta_trainer import train


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes:02d}min {secs:02d}s"
    if minutes > 0:
        return f"{minutes}min {secs:02d}s"
    return f"{secs}s"


def format_duration_hms(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def ensure_trials_csv(csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        return

    header = [
        "trial_id",
        "dataset",
        "split_mode",
        "fold_index",
        "lr",
        "batch_size",
        "dropout",
        "epochs",
        "train_rmse",
        "train_pearson",
        "val_rmse",
        "val_pearson",
        "test_rmse",
        "test_pearson",
        "test_mse",
        "test_mae",
        "checkpoint_path",
        "duration_seconds",
        "duration_hms",
        "status",
        "error_message",
        "debug_run",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(header)


def append_trial_result(csv_path: str, row: Dict[str, Any]) -> None:
    ensure_trials_csv(csv_path)

    ordered_row = [
        row.get("trial_id"),
        row.get("dataset"),
        row.get("split_mode"),
        row.get("fold_index"),
        row.get("lr"),
        row.get("batch_size"),
        row.get("dropout"),
        row.get("epochs"),
        row.get("train_rmse"),
        row.get("train_pearson"),
        row.get("val_rmse"),
        row.get("val_pearson"),
        row.get("test_rmse"),
        row.get("test_pearson"),
        row.get("test_mse"),
        row.get("test_mae"),
        row.get("checkpoint_path"),
        row.get("duration_seconds"),
        row.get("duration_hms"),
        row.get("status"),
        row.get("error_message"),
        row.get("debug_run"),
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(ordered_row)


def save_yaml(data: Dict[str, Any], yaml_path: str) -> None:
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    with open(yaml_path, "w", encoding="utf-8") as file:
        if yaml is not None:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
        else:
            json.dump(data, file, indent=4)


def is_better_result(candidate_metrics: Dict[str, Any], best_metrics: Optional[Dict[str, Any]]) -> bool:
    if best_metrics is None:
        return True

    candidate_rmse = candidate_metrics["val_rmse"]
    best_rmse = best_metrics["val_rmse"]

    if candidate_rmse < best_rmse:
        return True

    if candidate_rmse > best_rmse:
        return False

    return candidate_metrics["val_pearson"] > best_metrics["val_pearson"]


def generate_trials(
    lr_values: Iterable[float],
    batch_size_values: Iterable[int],
    dropout_values: Iterable[float],
) -> Iterable[Tuple[float, int, float]]:
    return itertools.product(lr_values, batch_size_values, dropout_values)


def build_best_config_payload(
    trial_id: int,
    trial_name: str,
    dataset_name: str,
    lr: float,
    batch_size: int,
    dropout: float,
    epochs: int,
    metrics: Dict[str, Any],
    model_dir: str,
    reports_dir: str,
) -> Dict[str, Any]:
    return {
        "model_name": "WideDTA",
        "status": "computed",
        "best_trial": {
            "trial_id": trial_id,
            "trial_name": trial_name,
            "dataset": dataset_name,
            "split_mode": metrics.get("split_mode"),
            "fold_index": metrics.get("fold_index"),
            "lr": lr,
            "batch_size": batch_size,
            "dropout": dropout,
            "epochs": epochs,
            "input_shapes": metrics.get("input_shapes"),
            "model_dir": model_dir,
            "reports_dir": reports_dir,
            "checkpoint_path": metrics.get("checkpoint_path"),
        },
        "best_metrics": {
            "train_rmse": metrics.get("train_rmse"),
            "train_pearson": metrics.get("train_pearson"),
            "val_rmse": metrics.get("val_rmse"),
            "val_pearson": metrics.get("val_pearson"),
            "test_rmse": metrics.get("test_rmse"),
            "test_pearson": metrics.get("test_pearson"),
            "test_mse": metrics.get("test_mse"),
            "test_mae": metrics.get("test_mae"),
        },
        "selection_rule": {
            "primary_metric": "val_rmse",
            "primary_goal": "min",
            "secondary_metric": "val_pearson",
            "secondary_goal": "max",
        },
    }


def run_hyperparameter_search(
    dataset_name: str,
    output_root: str,
    device: str | None,
    seed: int,
    epochs: int,
    lr_values: list[float],
    batch_size_values: list[int],
    dropout_values: list[float],
    val_split: float = 0.1,
    test_split: float = 0.2,
    max_train_batches: int | None = None,
    fold_index: int = 0,
    use_dataset_folds: bool = True,
) -> Dict[str, Any]:
    """
    @brief Run a grid search over WideDTA hyperparameters.
    """
    search_start_perf = time.perf_counter()
    search_start_wall = time.time()

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, run_name)
    models_dir = os.path.join(run_dir, "models")
    reports_dir = os.path.join(run_dir, "reports")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    trials_csv_path = os.path.join(reports_dir, "widedta_hyperparameter_trials.csv")
    best_config_yaml_path = os.path.join(reports_dir, "best_config_widedta.yaml")

    ensure_trials_csv(trials_csv_path)

    split_audit = audit_dataset_splits(
        dataset_name=dataset_name,
        fold_index=fold_index,
        use_dataset_folds=use_dataset_folds,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        output_dir=reports_dir,
    )
    if max_train_batches is not None:
        split_audit.setdefault("warnings", []).append("DEBUG RUN - metrics are not scientifically valid.")

    best_metrics: Dict[str, Any] | None = None
    best_trial_payload: Dict[str, Any] | None = None
    best_trial_name: str | None = None

    combinations = list(generate_trials(lr_values, batch_size_values, dropout_values))

    print(f"Total WideDTA trials to run: {len(combinations)}")
    print(f"Run directory: {run_dir}")
    print(f"Dataset={dataset_name}, use_dataset_folds={use_dataset_folds}, fold_index={fold_index}")

    for trial_index, (lr, batch_size, dropout) in enumerate(combinations, start=1):
        trial_name = f"trial_{trial_index:03d}"
        trial_models_dir = os.path.join(models_dir, trial_name)

        if os.path.exists(trial_models_dir):
            shutil.rmtree(trial_models_dir)

        os.makedirs(trial_models_dir, exist_ok=True)

        print(
            f"\n[{trial_name}] dataset={dataset_name}, lr={lr}, batch_size={batch_size}, "
            f"dropout={dropout}, epochs={epochs}, use_dataset_folds={use_dataset_folds}, fold_index={fold_index}"
        )

        trial_start_time = time.perf_counter()

        try:
            metrics = train(
                dataset_name=dataset_name,
                output_base=trial_models_dir,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                dropout=dropout,
                device=device,
                seed=seed,
                val_split=val_split,
                test_split=test_split,
                max_train_batches=max_train_batches,
                fold_index=fold_index,
                use_dataset_folds=use_dataset_folds,
            )

            trial_duration_seconds = round(time.perf_counter() - trial_start_time, 3)

            append_trial_result(
                trials_csv_path,
                {
                    "trial_id": trial_name,
                    "dataset": dataset_name,
                    "split_mode": metrics.get("split_mode"),
                    "fold_index": metrics.get("fold_index"),
                    "lr": lr,
                    "batch_size": batch_size,
                    "dropout": dropout,
                    "epochs": epochs,
                    "train_rmse": metrics.get("train_rmse"),
                    "train_pearson": metrics.get("train_pearson"),
                    "val_rmse": metrics.get("val_rmse"),
                    "val_pearson": metrics.get("val_pearson"),
                    "test_rmse": metrics.get("test_rmse"),
                    "test_pearson": metrics.get("test_pearson"),
                    "test_mse": metrics.get("test_mse"),
                    "test_mae": metrics.get("test_mae"),
                    "checkpoint_path": metrics.get("checkpoint_path"),
                    "duration_seconds": trial_duration_seconds,
                    "duration_hms": format_duration_hms(trial_duration_seconds),
                    "status": "success",
                    "error_message": "",
                    "debug_run": max_train_batches is not None,
                },
            )

            if is_better_result(metrics, best_metrics):
                best_metrics = metrics
                best_trial_name = trial_name
                best_trial_payload = build_best_config_payload(
                    trial_id=trial_index,
                    trial_name=trial_name,
                    dataset_name=dataset_name,
                    lr=lr,
                    batch_size=batch_size,
                    dropout=dropout,
                    epochs=epochs,
                    metrics=metrics,
                    model_dir=trial_models_dir,
                    reports_dir=reports_dir,
                )
                save_yaml(best_trial_payload, best_config_yaml_path)

        except Exception as exc:
            trial_duration_seconds = round(time.perf_counter() - trial_start_time, 3)

            append_trial_result(
                trials_csv_path,
                {
                    "trial_id": trial_name,
                    "dataset": dataset_name,
                    "split_mode": None,
                    "fold_index": fold_index if use_dataset_folds else None,
                    "lr": lr,
                    "batch_size": batch_size,
                    "dropout": dropout,
                    "epochs": epochs,
                    "train_rmse": None,
                    "train_pearson": None,
                    "val_rmse": None,
                    "val_pearson": None,
                    "test_rmse": None,
                    "test_pearson": None,
                    "test_mse": None,
                    "test_mae": None,
                    "checkpoint_path": "",
                    "duration_seconds": trial_duration_seconds,
                    "duration_hms": format_duration_hms(trial_duration_seconds),
                    "status": "failed_exception",
                    "error_message": str(exc)[:1000],
                    "debug_run": max_train_batches is not None,
                },
            )

            print(f"[WARNING] {trial_name} failed: {exc}")

    elapsed_seconds = round(time.perf_counter() - search_start_perf, 3)
    elapsed_time = format_duration(elapsed_seconds)
    search_elapsed_hms = format_duration_hms(time.time() - search_start_wall)

    if best_trial_payload is None:
        return {
            "status": "failed",
            "split_audit_json": split_audit.get("audit_path"),
            "message": "No valid WideDTA configuration found.",
            "run_dir": run_dir,
            "models_dir": models_dir,
            "reports_dir": reports_dir,
            "trials_csv": trials_csv_path,
            "best_config_yaml": best_config_yaml_path,
            "split_audit_json": split_audit.get("audit_path"),
            "elapsed_seconds": elapsed_seconds,
            "elapsed_time": elapsed_time,
            "hyperparameter_search_time": search_elapsed_hms,
        }

    best_trial_payload["hyperparameter_search_time"] = search_elapsed_hms
    save_yaml(best_trial_payload, best_config_yaml_path)

    return {
        "status": "success",
        "message": "WideDTA hyperparameter search completed successfully.",
        "best_trial": best_trial_name,
        "best_metrics": best_metrics,
        "run_dir": run_dir,
        "models_dir": models_dir,
        "reports_dir": reports_dir,
        "trials_csv": trials_csv_path,
        "best_config_yaml": best_config_yaml_path,
        "split_audit_json": split_audit.get("audit_path"),
        "debug_run": max_train_batches is not None,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_time": elapsed_time,
        "hyperparameter_search_time": search_elapsed_hms,
    }
