import ast
import csv
import random
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
    output_dir = Path(save_directory) / "CheapNet_hyperparameter_tuning_all_trials"
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
