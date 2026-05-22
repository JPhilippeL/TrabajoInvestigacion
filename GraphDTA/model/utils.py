import ast
import os
import random

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset

from GraphDTA.GNN_model.gat import GATNet
from GraphDTA.GNN_model.gat_gcn import GAT_GCN
from GraphDTA.GNN_model.gcn import GCNNet
from GraphDTA.GNN_model.ginconv import GINConvNet


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


def val(model, dataloader, device):
    model.eval()
    pred_list, label_list = [], []

    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).view(-1)
            target = data.y.view(-1).float()

        pred_list.append(pred.cpu().numpy())
        label_list.append(target.cpu().numpy())

    pred = np.concatenate(pred_list)
    label = np.concatenate(label_list)

    rmse = np.sqrt(mean_squared_error(label, pred))
    pearson = np.corrcoef(pred, label)[0, 1]

    model.train()
    return rmse, pearson


def initialize_model(model_name, n_filters, drop_out):
    model = None
    if model_name == "GAT":
        model = GATNet(n_filters=n_filters, dropout=drop_out)
    elif model_name == "GAT_GCN":
        model = GAT_GCN(n_filters=n_filters, dropout=drop_out)
    elif model_name == "GINConvNet":
        model = GINConvNet(n_filters=n_filters, dropout=drop_out)
    elif model_name == "GCN":
        model = GCNNet(n_filters=n_filters, dropout=drop_out)

    if model is None:
        raise Exception(f"Model {model_name} not found.")
    return model


def load_split_txt(path):
    with open(path) as f:
        return ast.literal_eval(f.read())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
