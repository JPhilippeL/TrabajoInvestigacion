"""
My work in this file are based on :
@reference : https://docs.pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html by PyTorch
@description : perform hyperparameter tuning of the model GIGN
"""

import json
import os
import ast
import random

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

import ray
from ray import tune
from ray.air import RunConfig, CheckpointConfig

from GIGN_GUI.model.GIGN_model import GIGN


# =========================================================
# Utils
# =========================================================


def load_split_txt(path):
    with open(path, "r") as f:
        return ast.literal_eval(f.read())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def val(model, dataloader, device):
    model.eval()
    pred_list, label_list = [], []

    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)

        pred_list.append(pred.view(-1).cpu().numpy())
        label_list.append(data.y.view(-1).cpu().numpy())

    pred = np.concatenate(pred_list)
    label = np.concatenate(label_list)

    rmse = np.sqrt(mean_squared_error(label, pred))
    pearson = np.corrcoef(pred, label)[0, 1]

    model.train()
    return rmse, pearson


def save_tuning_in_a_file(results, file):
    best_result = results.get_best_result(metric="mean_val_rmse", mode="min")
    df = results.get_dataframe()

    with open(file, "w", encoding="UTF-8") as f:
        f.write("GIGN Hyperparameter Tuning Results\n")
        f.write("//BEST CONFIG:\n")
        f.write(json.dumps(best_result.config, indent=4, ensure_ascii=False))
        f.write("\n\n")

        f.write("//METRIC:\n")
        best_metrics = {
            "mean_val_rmse": best_result.metrics.get("mean_val_rmse"),
            "std_val_rmse": best_result.metrics.get("std_val_rmse"),
            "mean_val_pearson": best_result.metrics.get("mean_val_pearson"),
            "std_val_pearson": best_result.metrics.get("std_val_pearson"),
        }
        f.write(json.dumps(best_metrics, indent=4, ensure_ascii=False))
        f.write("\n\n")

        f.write("//ALL CONFIG:\n")
        # Output files in Ray Tune starts with config
        config_cols = [c for c in df.columns if c.startswith("config/")]

        for _, row in df.iterrows():
            config_dict = {}
            for col in config_cols:
                key = col.replace("config/", "")
                value = row[col]

                if hasattr(value, "item"):
                    value = value.item()

                config_dict[key] = value

            f.write(json.dumps(config_dict, ensure_ascii=False))
            f.write("\n")


def is_finite_number(number):
    return number is not None and np.isfinite(number)


# to avoid NAN and INF
def safe_mean_std(values, empty_mean=float("inf"), empty_std=float("inf")):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return empty_mean, empty_std

    return float(np.mean(arr)), float(np.std(arr))


# =========================================================
# Dataset
# =========================================================


class URVGraphDataset(Dataset):
    def __init__(self, graph_dir, pdb_ids):
        self.graph_dir = graph_dir
        self.pdb_ids = pdb_ids

    def __len__(self):
        return len(self.pdb_ids)

    def __getitem__(self, idx):
        pdb_id = self.pdb_ids[idx]
        path = os.path.join(self.graph_dir, f"{pdb_id}.pt")
        data = torch.load(path, weights_only=False)
        return data


def load_data_for_split(
        split_id, batch_size, train_splits, val_splits, test_splits, graph_dir
):
    train_ids = train_splits[split_id]
    val_ids = val_splits[split_id]
    test_ids = test_splits[split_id]

    train_set = URVGraphDataset(graph_dir, train_ids)
    val_set = URVGraphDataset(graph_dir, val_ids)
    test_set = URVGraphDataset(graph_dir, test_ids)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# =========================================================
# Train on one split
# =========================================================
def train_on_one_split(
        config, split_id, device, train_splits, test_splits, val_splits, graph_dir
):
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
                print(
                    f"[WARNING] Loss not finished yet on split {split_id}, epoch {epoch}"
                )
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


# =========================================================
# Train
# =========================================================


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
            config, split_id, device, train_splits, test_splits, val_splits, graph_dir
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
    mean_val_pearson, std_val_pearson = safe_mean_std(
        val_pearsons, empty_mean=0.0, empty_std=0.0
    )
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


def HyperParameter_tuning(
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "NODE_DIM": 14,
        "HIDDEN_DIM": tune.choice([32, 64, 128, 256]),
        "BATCH_SIZE": tune.choice([4, 8]),
        "LR": tune.loguniform(1e-4, 1e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "EPOCHS": 50,
        "patience": 15,
        "drop_out": tune.choice([0, 0.05, 0.1])
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
                num_to_keep=0,
                checkpoint_at_end=False,
            ),
        ),

    )
    results = tuner.fit()

    best_result = results.get_best_result(metric="mean_val_rmse", mode="min")
    print(f"Best result {best_result}")
    save_tuning_in_a_file(results, out_dir)
