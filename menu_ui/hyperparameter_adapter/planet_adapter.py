import json
import logging
import sys
import time

from PySide6.QtWidgets import QDialog
from ray import tune

from core.hyperparameter_tune_script.planet_hp import run_hyperparameter_search

logger = logging.getLogger(__name__)


def dialog_accepted(dialog):
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.get_inputs()

    return None


def pop_unused_key(dialog_inputs):
    config = dialog_inputs.copy()

    unused_keys = [
        "data_output_path",
        "save_directory",
        "cpu_per_trials",
        "gpu_per_trials",
        "number_of_trials",
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


def create_ray_tune_config(dialog_inputs):
    config = pop_unused_key(dialog_inputs)

    batch_size_interval = create_interval_by_multiplication(
        config["batch_size_min"],
        config["batch_size_max"],
    )

    feature_dims_interval = create_interval_by_multiplication(
        config["feature_dims_min"],
        config["feature_dims_max"],
    )

    key_dims_interval = create_interval_by_multiplication(
        config["key_dims_min"],
        config["key_dims_max"],
    )

    value_dims_interval = create_interval_by_multiplication(
        config["value_dims_min"],
        config["value_dims_max"],
    )

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
        "epochs": config["EPOCHS"],
        "patience": config["PATIENCE"],
        "seed": config["seed"],
        "num_workers": config["num_workers"],
        "feature_dims": tune.choice(feature_dims_interval),
        "nheads": tune.choice(
            list(
                range(
                    config["nheads_min"],
                    config["nheads_max"] + 1,
                )
            )
        ),
        "key_dims": tune.choice(key_dims_interval),
        "value_dims": tune.choice(value_dims_interval),
        "pro_update_inters": tune.choice(
            list(
                range(
                    config["pro_update_inters_min"],
                    config["pro_update_inters_max"] + 1,
                )
            )
        ),
        "lig_update_iters": tune.choice(
            list(
                range(
                    config["lig_update_iters_min"],
                    config["lig_update_iters_max"] + 1,
                )
            )
        ),
        "pro_lig_update_iters": tune.choice(
            list(
                range(
                    config["pro_lig_update_iters_min"],
                    config["pro_lig_update_iters_max"] + 1,
                )
            )
        ),
        "clip_norm": tune.uniform(
            dialog_inputs["clip_norm_min"],
            dialog_inputs["clip_norm_max"],
        ),
        "beta_start_step": tune.choice(
            list(
                range(
                    config["beta_start_step_min"],
                    config["beta_start_step_max"] + 1,
                    config["beta_start_step_step"],
                )
            )
        ),
    }


def launch_ray_tune_hyperparameter_search(dialog_inputs, log_callback):
    config = create_ray_tune_config(dialog_inputs)

    run_hyperparameter_search(
        save_directory=dialog_inputs["save_directory"],
        data_output_path=dialog_inputs["data_output_path"],
        search_space=config,
        cpu_used_per_trials=dialog_inputs["cpu_per_trials"],
        gpu_used_per_trials=dialog_inputs["gpu_per_trials"],
        number_of_trials=dialog_inputs["number_of_trials"],
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

    logger.info(f"Time elapsed for PLANET hyperparameter tuning is: {elapsed_time:.2f} seconds")
