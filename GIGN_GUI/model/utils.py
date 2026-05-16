import ast
import os
import random

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset


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
        f.write(f"drop_rate: {drop_out}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"weight_decay: {weight_decay}\n")
        f.write(f"patience: {patience}\n")




def escala_global(file_path):
    labels = np.loadtxt(file_path, usecols=1)

    real_min = labels.min()
    real_max = labels.max()

    margin = 0.2
    global_min = real_min - margin
    global_max = real_max + margin

    print("Global axis limits:", global_min, global_max)

    np.save("global_axis.npy", [global_min, global_max])

    return global_min, global_max
