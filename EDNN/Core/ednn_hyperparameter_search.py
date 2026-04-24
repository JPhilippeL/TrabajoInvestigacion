"""
@file ednn_hyperparameter_search.py
@author Mohamed EL BOUKHIARI
@brief Hyperparameter search pipeline for the EDNN module.
"""

from __future__ import annotations

import csv
import itertools
import os
import shutil
from typing import Any, Dict, Iterable, Optional, Tuple

import yaml

from EDNN.Core.ednn_trainer import train
from EDNN.Core.ednn_tester import test_model


def ensure_trials_csv(csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        return

    header = [
        "trial_id",
        "lr",
        "hidden_dim",
        "batch_size",
        "rmse_mean",
        "pearson_mean",
        "spearman_mean",
        "status",
        "error_message",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_trial_result(csv_path: str, row: Dict[str, Any]) -> None:
    ensure_trials_csv(csv_path)

    ordered_row = [
        row.get("trial_id"),
        row.get("lr"),
        row.get("hidden_dim"),
        row.get("batch_size"),
        row.get("rmse_mean"),
        row.get("pearson_mean"),
        row.get("spearman_mean"),
        row.get("status"),
        row.get("error_message"),
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ordered_row)


def save_yaml(data: Dict[str, Any], yaml_path: str) -> None:
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def build_best_config_payload(
    trial_id: int,
    lr: float,
    hidden_dim: int,
    batch_size: int,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "model_name": "EDNN",
        "status": "computed",
        "best_trial": {
            "trial_id": trial_id,
            "lr": lr,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
        },
        "best_metrics": {
            "rmse_mean": metrics.get("RMSE"),
            "pearson_mean": metrics.get("Pearson"),
            "spearman_mean": metrics.get("Spearman"),
        },
        "selection_rule": {
            "primary_metric": "RMSE",
            "primary_goal": "min",
            "secondary_metric": "Pearson",
            "secondary_goal": "max",
        },
    }


def is_better_result(
    candidate_metrics: Dict[str, Any],
    best_metrics: Optional[Dict[str, Any]],
) -> bool:
    if best_metrics is None:
        return True

    candidate_rmse = candidate_metrics["RMSE"]
    best_rmse = best_metrics["RMSE"]

    if candidate_rmse < best_rmse:
        return True
    if candidate_rmse > best_rmse:
        return False

    candidate_pearson = candidate_metrics["Pearson"]
    best_pearson = best_metrics["Pearson"]

    return candidate_pearson > best_pearson


def generate_trials(
    lr_values: Iterable[float],
    hidden_dim_values: Iterable[int],
    batch_size_values: Iterable[int],
) -> Iterable[Tuple[float, int, int]]:
    return itertools.product(lr_values, hidden_dim_values, batch_size_values)


def run_hyperparameter_search(
    graphs_dir: str,
    train_split_file: str,
    val_split_file: str,
    test_split_file: str,
    models_root: str,
    results_root: str,
    temp_runs_dir: str,
    device: str | None,
    seed: int,
    epochs: int,
    patience: int,
    lr_values: list[float],
    hidden_dim_values: list[int],
    batch_size_values: list[int],
) -> Dict[str, Any]:
    os.makedirs(temp_runs_dir, exist_ok=True)
    os.makedirs(models_root, exist_ok=True)
    os.makedirs(results_root, exist_ok=True)

    trials_csv_path = os.path.join(results_root, "ednn_hyperparameter_trials.csv")
    best_config_yaml_path = os.path.join(results_root, "best_config_ednn.yaml")

    ensure_trials_csv(trials_csv_path)

    best_metrics: Dict[str, Any] | None = None
    best_trial_payload: Dict[str, Any] | None = None
    best_trial_name: str | None = None

    combinations = list(generate_trials(lr_values, hidden_dim_values, batch_size_values))

    print(f"Total trials to run: {len(combinations)}")

    for trial_index, (lr, hidden_dim, batch_size) in enumerate(combinations, start=1):
        trial_name = f"trial_{trial_index:03d}"
        trial_root = os.path.join(temp_runs_dir, trial_name)
        trial_models_dir = os.path.join(trial_root, "Models_EDNN")
        trial_results_dir = os.path.join(trial_root, "Results_EDNN")

        if os.path.exists(trial_root):
            shutil.rmtree(trial_root)

        os.makedirs(trial_models_dir, exist_ok=True)
        os.makedirs(trial_results_dir, exist_ok=True)

        print(f"\n[{trial_name}] lr={lr}, hidden_dim={hidden_dim}, batch_size={batch_size}")

        try:
            train(
                graphs_dir=graphs_dir,
                train_split_file=train_split_file,
                val_split_file=val_split_file,
                test_split_file=test_split_file,
                output_base=trial_models_dir,
                batch_size=batch_size,
                epochs=epochs,
                patience=patience,
                lr=lr,
                hidden_dim=hidden_dim,
                device=device,
                seed=seed,
            )

            metrics = test_model(
                graphs_dir=graphs_dir,
                test_split_file=test_split_file,
                models_dir=trial_models_dir,
                results_dir=trial_results_dir,
                batch_size=batch_size,
                device=device,
                hidden_dim=hidden_dim,
            )

            append_trial_result(
                trials_csv_path,
                {
                    "trial_id": trial_name,
                    "lr": lr,
                    "hidden_dim": hidden_dim,
                    "batch_size": batch_size,
                    "rmse_mean": metrics["RMSE"],
                    "pearson_mean": metrics["Pearson"],
                    "spearman_mean": metrics["Spearman"],
                    "status": "success",
                    "error_message": "",
                },
            )

            if is_better_result(metrics, best_metrics):
                best_metrics = metrics
                best_trial_name = trial_name
                best_trial_payload = build_best_config_payload(
                    trial_id=trial_index,
                    lr=lr,
                    hidden_dim=hidden_dim,
                    batch_size=batch_size,
                    metrics=metrics,
                )
                save_yaml(best_trial_payload, best_config_yaml_path)

                if os.path.exists(models_root):
                    best_models_dir = os.path.join(models_root, "best_trial_models")
                    if os.path.exists(best_models_dir):
                        shutil.rmtree(best_models_dir)
                    shutil.copytree(trial_models_dir, best_models_dir)

        except Exception as exc:
            append_trial_result(
                trials_csv_path,
                {
                    "trial_id": trial_name,
                    "lr": lr,
                    "hidden_dim": hidden_dim,
                    "batch_size": batch_size,
                    "rmse_mean": None,
                    "pearson_mean": None,
                    "spearman_mean": None,
                    "status": "failed_exception",
                    "error_message": str(exc)[:1000],
                },
            )

    if best_trial_payload is None:
        return {
            "status": "failed",
            "message": "No valid configuration found.",
            "trials_csv": trials_csv_path,
            "best_config_yaml": best_config_yaml_path,
        }

    return {
        "status": "success",
        "message": "Hyperparameter search completed successfully.",
        "best_trial": best_trial_name,
        "best_metrics": best_metrics,
        "trials_csv": trials_csv_path,
        "best_config_yaml": best_config_yaml_path,
        "best_models_dir": os.path.join(models_root, "best_trial_models"),
    }
