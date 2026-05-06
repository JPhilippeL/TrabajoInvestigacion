"""
My work in this file are based on :
@reference : https://docs.pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html by PyTorch
@description : perform hyperparameter tuning of the model GIGN
"""

import numpy as np
import ray
import torch
from ray import tune
from ray.air import CheckpointConfig

from GIGN_GUI.model.gign_model import GIGN
from GIGN_GUI.model.utils import (
    is_finite_number,
    load_data_for_split,
    load_split_txt,
    safe_mean_std,
    save_tuning_in_a_file,
    seed_everything,
    val,
)


def train_on_one_split(config, split_id, device, train_splits, test_splits, val_splits, graph_dir):
    seed_everything(split_id)

    train_loader, val_loader, test_loader = load_data_for_split(
        split_id=split_id,
        batch_size=config["BATCH_SIZE"],
        train_splits=train_splits,
        val_splits=val_splits,
        test_splits=test_splits,
        graph_dir=graph_dir,
    )

    model = GIGN(config["NODE_DIM"], config["HIDDEN_DIM"], config["drop_out"])
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["LR"], weight_decay=config["weight_decay"]
    )
    criterion = torch.nn.MSELoss()

    best_val_rmse = float("inf")
    best_val_pearson = -1.0

    patience_counter = 0
    for epoch in range(config["EPOCHS"]):
        model.train()
        epoch_loss = 0.0
        samples = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data).view(-1)
            target = data.y.view(-1)
            loss = criterion(pred, target)
            if not torch.isfinite(loss).item():
                print(f"[WARNING] Loss not finished yet on split {split_id}, epoch {epoch}")
                continue
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * target.size(0)
            samples += target.size(0)

        if samples == 0:
            train_rmse = float("inf")
        else:
            train_rmse = np.sqrt(epoch_loss / samples)
        val_rmse, val_pearson = val(model, val_loader, device)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_val_pearson = val_pearson
            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            break

    return {
        "best_val_rmse": best_val_rmse,
        "best_val_pearson": best_val_pearson,
        "last_train_rmse": train_rmse,
    }


def train_GIGN(config, train_split_file, val_split_file, test_split_file, graph_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_splits = load_split_txt(train_split_file)
    val_splits = load_split_txt(val_split_file)
    test_splits = load_split_txt(test_split_file)

    val_rmses = []
    val_pearsons = []
    train_rmses = []

    num_splits = len(train_splits)

    for split_id in range(num_splits):
        metrics = train_on_one_split(
            config,
            split_id,
            device,
            train_splits,
            test_splits,
            val_splits,
            graph_dir,
        )

        if is_finite_number(metrics.get("best_val_rmse")):
            val_rmses.append(metrics["best_val_rmse"])

        if is_finite_number(metrics.get("best_val_pearson")):
            val_pearsons.append(metrics["best_val_pearson"])

        if is_finite_number(metrics.get("last_train_rmse")):
            train_rmses.append(metrics["last_train_rmse"])

    mean_val_rmse, std_val_rmse = safe_mean_std(
        val_rmses, empty_mean=float("inf"), empty_std=float("inf")
    )
    mean_val_pearson, std_val_pearson = safe_mean_std(val_pearsons, empty_mean=0.0, empty_std=0.0)
    mean_train_rmse, std_train_rmse = safe_mean_std(
        train_rmses, empty_mean=float("inf"), empty_std=float("inf")
    )
    tune.report(
        {
            "mean_train_rmse": mean_train_rmse,
            "std_train_rmse": std_train_rmse,
            "mean_val_rmse": mean_val_rmse,
            "std_val_rmse": std_val_rmse,
            "mean_val_pearson": mean_val_pearson,
            "std_val_pearson": std_val_pearson,
        }
    )


def hyperparameter_tuning(
    cpu_per_trials,
    gpu_per_trials,
    num_trials,
    out_dir,
    train_split_file,
    val_split_file,
    test_split_file,
    graph_dir,
):
    ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    config = {
        "NODE_DIM": 14,
        "HIDDEN_DIM": tune.choice([32, 64, 128, 256]),
        "BATCH_SIZE": tune.choice([4, 8]),
        "LR": tune.loguniform(1e-4, 1e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "EPOCHS": 50,
        "patience": 15,
        "drop_out": tune.choice([0, 0.05, 0.1]),
    }

    trainable = tune.with_parameters(
        train_GIGN,
        train_split_file=train_split_file,
        val_split_file=val_split_file,
        test_split_file=test_split_file,
        graph_dir=graph_dir,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            trainable,
            resources={"cpu": cpu_per_trials, "gpu": gpu_per_trials},
        ),
        tune_config=tune.TuneConfig(
            metric="mean_val_rmse",
            mode="min",
            num_samples=num_trials,
        ),
        param_space=config,
        run_config=ray.air.RunConfig(
            name="GIGN_hyperparameter_tuning",
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=False,
            ),
        ),
    )
    results = tuner.fit()

    best_result = results.get_best_result(metric="mean_val_rmse", mode="min")
    print(f"Best result {best_result}")
    save_tuning_in_a_file(results, out_dir)


if __name__ == "__main__":
    graph_dir = "/home/administrateur/Bureau/deepGNN/GIGN/Graphs_GIGN"
    output_dir = "tuning"
    test_split_file = (
        "/home/administrateur/Bureau/deepGNN/MPro-URV_Version2/Splits/test_index_folder.txt"
    )
    train_split_file = (
        "/home/administrateur/Bureau/deepGNN/MPro-URV_Version2/Splits/train_index_folder.txt"
    )
    val_split_file = (
        "/home/administrateur/Bureau/deepGNN/MPro-URV_Version2/Splits/val_index_folder.txt"
    )
    cpu_per_trials = 4
    gpu_per_trials = 0
    num_trials = 20

    hyperparameter_tuning(
        cpu_per_trials,
        gpu_per_trials,
        num_trials,
        output_dir,
        train_split_file,
        val_split_file,
        test_split_file,
        graph_dir,
    )
