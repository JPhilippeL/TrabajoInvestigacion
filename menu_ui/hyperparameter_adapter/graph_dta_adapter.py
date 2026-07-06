import json
import logging
import sys
import time

from PySide6.QtWidgets import QDialog
from ray import tune

from core.hyperparameter_tune_script.graph_dta_hp import launch_hyperparametre_search

logger = logging.getLogger(__name__)


def dialog_accepted(dialog):
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.get_inputs()

    return None


def pop_unused_key(dialog_inputs):
    config = dialog_inputs.copy()

    unused_keys = [
        "train_split_file",
        "val_split_file",
        "test_split_file",
        "output_dir",
        "graph_directory",
        "cpu_per_trial",
        "gpu_per_trial",
        "num_samples",
        "graphdta_model_name",
        "model_name",
        "PATIENCE",
    ]

    for key in unused_keys:
        config.pop(key, None)

    return config


def create_interval_by_multiplication(min_value, max_value):
    values = []

    current = min_value
    while current <= max_value:
        values.append(current)
        current *= 2

    if not values:
        values.append(min_value)

    return values


def create_interval_dropout(min_value, max_value):
    values = []

    current = min_value
    while current <= max_value:
        values.append(round(current, 2))
        current += 0.05

    if not values:
        values.append(min_value)

    return values


def create_ray_tune_config(dialog_inputs):
    config = pop_unused_key(dialog_inputs)

    n_filters_interval = create_interval_by_multiplication(
        config["n_filters_min"],
        config["n_filters_max"],
    )

    batch_size_interval = create_interval_by_multiplication(
        config["batch_size_min"],
        config["batch_size_max"],
    )

    dropout_interval = create_interval_dropout(config["dropout_min"], config["dropout_max"])

    return {
        "batch_size": tune.choice(batch_size_interval),
        "lr": tune.loguniform(
            dialog_inputs["lr_min"],
            dialog_inputs["lr_max"],
        ),
        "weight_decay": tune.loguniform(
            dialog_inputs["weight_decay_min"],
            dialog_inputs["weight_decay_max"],
        ),
        "dropout": tune.choice(dropout_interval),
        "n_filters": tune.choice(n_filters_interval),
        "epochs": config["EPOCHS"],
    }


def launch_ray_tune_hyperparameter_search(dialog_inputs, log_callback):
    config = create_ray_tune_config(dialog_inputs)

    launch_hyperparametre_search(
        config=config,
        cpu_per_trial=dialog_inputs["cpu_per_trial"],
        gpu_per_trial=dialog_inputs["gpu_per_trial"],
        num_samples=dialog_inputs["num_samples"],
        output_dir=dialog_inputs["output_dir"],
        model_name=dialog_inputs["graphdta_model_name"],
        train_split_file=dialog_inputs["train_split_file"],
        val_split_file=dialog_inputs["val_split_file"],
        test_split_file=dialog_inputs["test_split_file"],
        graph_dir=dialog_inputs["graph_directory"],
        log_callback=log_callback,
    )


if __name__ == "__main__":
    start_hp = time.time()

    params = json.loads(sys.argv[1])

    launch_ray_tune_hyperparameter_search(
        params,
        log_callback=logger,
    )

    end_hp = time.time()
    elapsed_time = end_hp - start_hp

    logger.info(f"Time elapsed for GraphDTA hyperparameter tuning is: {elapsed_time:.2f} seconds")
