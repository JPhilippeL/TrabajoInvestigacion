import json
import logging
import sys

from PySide6.QtWidgets import QDialog
from ray import tune

from URVDEEPTAF.hyperparameter_search.urvdtaf_hyperparameter_search import (
    run_hyperparameter_search,
)
from URVDEEPTAF.ui.dialogs.hp_tuning_dialog import (
    DeepDTAFHyperparameterSearchDialog,
)

logger = logging.getLogger(__name__)


def open_dialog(parent=None):
    dialog = DeepDTAFHyperparameterSearchDialog(parent)

    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.get_inputs()

    return None


def partition_batch_size(min_batch_size, max_batch_size):
    if min_batch_size <= 0:
        raise ValueError("Minimum batch size must be greater than 0.")

    if max_batch_size < min_batch_size:
        raise ValueError("Maximum batch size must be greater than or equal to minimum batch size.")

    batch_size = min_batch_size
    batch_size_list = []

    while batch_size <= max_batch_size:
        batch_size_list.append(batch_size)
        batch_size *= 2

    return batch_size_list


def config(params):
    batch_sizes = partition_batch_size(
        params["batch_size_min"],
        params["batch_size_max"],
    )

    return {
        "lr": tune.loguniform(
            params["lr_min"],
            params["lr_max"],
        ),
        "weight_decay": tune.loguniform(
            params["weight_decay_min"],
            params["weight_decay_max"],
        ),
        "batch_size": tune.choice(batch_sizes),
        "epochs": params["epochs"],
        "patience": params["patience"],
        "max_seq_len": params["max_seq_len"],
        "max_pkt_len": params["max_pkt_len"],
        "max_smi_len": params["max_smi_len"],
    }


def launch_tuning(params):
    search_space = config(params)

    return run_hyperparameter_search(
        model_name=params["model_name"],
        mother_dir=params["data_directory"],
        output_path=params["output_directory"],
        search_space=search_space,
        cpu_used_per_trial=params["cpu_per_trial"],
        gpu_used_per_trial=params["gpu_per_trial"],
        number_of_trials=params["number_of_trials"],
        seed=params["seed"],
        num_workers=params["num_workers"],
        save_best_epoch=params["save_best_epoch"],
        log_callback=lambda message: print(message, flush=True),
    )


if __name__ == "__main__":
    params = json.loads(sys.argv[1])
    launch_tuning(params)