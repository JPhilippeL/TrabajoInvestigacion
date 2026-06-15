import json
import logging
import sys
import time

from PySide6.QtWidgets import QDialog
from ray import tune

from core.hyperparameter_tune_script.script.cheapnet_hp import run_hyperparameter_search

logger = logging.getLogger(__name__)


def dialog_accepted(dialog):
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.get_inputs()

    return None


def pop_unused_key(dialog_inputs):
    config = dialog_inputs.copy()
    unused_keys = [
        "train_file",
        "test_file",
        "val_file",
        "save_directory",
        "graph_directory",
        "cpu_per_trials",
        "gpu_per_trials",
        "number_of_trials",
    ]
    for key in unused_keys:
        config.pop(key, None)

    return config


def create_ray_tune_config(dialog_inputs):
    config = pop_unused_key(dialog_inputs)
    hidden_dim_interval = []
    hidden_dim_min = config["hidden_dim_min"]
    while hidden_dim_min <= config["hidden_dim_max"]:
        hidden_dim_interval.append(hidden_dim_min)
        hidden_dim_min = hidden_dim_min * 2

    batch_size_interval = []
    batch_size_min = config["batch_size_min"]
    while batch_size_min <= config["batch_size_max"]:
        batch_size_interval.append(batch_size_min)
        batch_size_min = batch_size_min * 2

    drop_out_interval = []
    drop_out_min = config["drop_out_min"]
    while drop_out_min <= config["drop_out_max"]:
        drop_out_interval.append(drop_out_min)
        drop_out_min = round(drop_out_min + 0.05, 10)
    return {
        "NODE_DIM": config["NODE_DIM"],
        "hidden_dim": tune.choice(hidden_dim_interval),
        "batch_size": tune.choice(batch_size_interval),
        "lr": tune.loguniform(dialog_inputs["lr_min"], dialog_inputs["lr_max"]),
        "weight_decay": tune.loguniform(
            dialog_inputs["weight_decay_min"], dialog_inputs["weight_decay_max"]
        ),
        "EPOCHS": config["EPOCHS"],
        "PATIENCE": config["PATIENCE"],
        "drop_out": tune.choice(drop_out_interval),
    }


def launch_ray_tune_hyperparameter_search(dialog_input, log_callback):
    config = create_ray_tune_config(dialog_input)
    run_hyperparameter_search(
        dialog_input["save_directory"],
        dialog_input["train_file"],
        dialog_input["test_file"],
        dialog_input["val_file"],
        config,
        dialog_input["graph_directory"],
        dialog_input["cpu_per_trials"],
        dialog_input["gpu_per_trials"],
        dialog_input["number_of_trials"],
        log_callback=log_callback,
    )


if __name__ == "__main__":
    start_hp = time.time()
    params = json.loads(sys.argv[1])
    launch_ray_tune_hyperparameter_search(params, log_callback=logger.info)
    end_hp = time.time()
    elapsed_time = end_hp - start_hp
    logger.info(f"Time elapsed for hyperparameter tuning is: {elapsed_time} seconds ")
