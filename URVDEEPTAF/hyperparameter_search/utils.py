import csv
from pathlib import Path



def save_split_results_csv(split_results, trial_dir):
    csv_path = Path(trial_dir) / "split_results.csv"

    fieldnames = [
        "split_id",
        "split_path",
        "run_dir",
        "best_epoch",
        "best_train_rmse",
        "best_val_rmse",
        "test_rmse",
        "test_pearson",
        "best_model_path",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_results)

    return csv_path


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

        row["mean_train_rmse"] = metrics.get("mean_train_rmse")
        row["std_train_rmse"] = metrics.get("std_train_rmse")

        row["mean_val_rmse"] = metrics.get("mean_val_rmse")
        row["std_val_rmse"] = metrics.get("std_val_rmse")

        row["mean_test_rmse"] = metrics.get("mean_test_rmse")
        row["std_test_rmse"] = metrics.get("std_test_rmse")

        row["mean_test_pearson"] = metrics.get("mean_test_pearson")
        row["std_test_pearson"] = metrics.get("std_test_pearson")

        rows.append(row)

    if not rows:
        return csv_path

    fieldnames = sorted(rows[0].keys())

    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path
