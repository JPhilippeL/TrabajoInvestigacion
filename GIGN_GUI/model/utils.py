import ast
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset


def load_split_txt(path):
    with open(path) as f:
        return ast.literal_eval(f.read())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def val(model, dataloader, device):
    model.eval()
    pred_list, label_list = [], []

    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)

        pred_list.append(pred.cpu().numpy())
        label_list.append(data.y.cpu().numpy())

    pred = np.concatenate(pred_list)
    label = np.concatenate(label_list)

    rmse = np.sqrt(mean_squared_error(label, pred))
    pearson = np.corrcoef(pred, label)[0, 1]

    model.train()
    return rmse, pearson


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


def write_hyperparameter_into_a_file(
    epochs,
    node_dim,
    hidden_dim,
    drop_out,
    batch_size,
    lr,
    weight_decay,
    patience,
    output_file,
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True) if os.path.dirname(
        output_file
    ) else None
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Hyperparameters:\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"node_dim: {node_dim}\n")
        f.write(f"hidden_dim: {hidden_dim}\n")
        f.write(f"drop_out: {drop_out}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"weight_decay: {weight_decay}\n")
        f.write(f"patience: {patience}\n")


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


def is_finite_number(number):
    return number is not None and np.isfinite(number)


# to avoid NAN and INF
def safe_mean_std(values, empty_mean=float("inf"), empty_std=float("inf")):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return empty_mean, empty_std

    return float(np.mean(arr)), float(np.std(arr))


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
