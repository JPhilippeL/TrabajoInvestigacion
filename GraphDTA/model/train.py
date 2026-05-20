import ast
import os
import random

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from GraphDTA.GNN_model.gat import GATNet
from GraphDTA.GNN_model.gat_gcn import GAT_GCN
from GraphDTA.GNN_model.gcn import GCNNet
from GraphDTA.GNN_model.ginconv import GINConvNet


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


def initialize_model(model_name):
    if model_name == "GAT":
        model = GATNet()
    elif model_name == "GAT_GCN":
        model = GAT_GCN()
    elif model_name == "GINConvNet":
        model = GINConvNet()
    elif model_name == "GCNNet":
        model = GCNNet()

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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "./GraphDTA/MODEL"
    os.makedirs(output_dir, exist_ok=True)
    train_split_file = "/home/administrateur/Bureau/deepGNN/MPro-URV_Version2/Splits/train_index_folder.txt"
    test_split_file = "/home/administrateur/Bureau/deepGNN/MPro-URV_Version2/Splits/test_index_folder.txt"
    val_split_file = "/home/administrateur/Bureau/deepGNN/MPro-URV_Version2/Splits/valid_index_folder.txt"
    graph_dir = "/home/administrateur/Bureau/TrabajoInvestigacion/GraphDTA/graph"
    train_split = load_split_txt(train_split_file)
    val_split = load_split_txt(val_split_file)
    test_split = load_split_txt(test_split_file)

    split_best_val_rmses = []
    split_test_rmses = []
    split_test_pearsons = []

    batch_size = 8
    for split_id in range(len(train_split)):
        print(f"SPLIT: {split_id}")

        seed_everything(split_id + 42)

        train_ids = train_split[split_id]
        test_ids = test_split[split_id]
        val_ids = val_split[split_id]

        train_set = URVGraphDataset(graph_dir, train_ids)
        test_set = URVGraphDataset(graph_dir, test_ids)
        val_set = URVGraphDataset(graph_dir, val_ids)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # use default parameter of the model
        model = initialize_model("GAT")
        model.to(device)
        print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_rmse = float("inf")
        best_epoch = -1
        split_save_dir = os.path.join(output_dir, f"split_{split_id:02}")
        os.makedirs(split_save_dir, exist_ok=True)
        best_model_path = os.path.join(split_save_dir, "best_model.pt")

        for epoch in range(50):
            model.train()
            epoch_loss = 0.0
            n_train = 0
            for data in train_loader:
                data = data.to(device)
                pred = model(data)
                target = data.y.view(-1, 1).float()
                loss = loss_fn(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size_current = target.size(0)
                epoch_loss += loss.item() * batch_size_current
                n_train += batch_size_current

            train_rmse = np.sqrt(epoch_loss / n_train)
            val_rmse, val_pr = val(model, val_loader, device)

            print(
                f"Epoch {epoch:03d} | "
                f"Train RMSE: {train_rmse:.4f} | "
                f"Val RMSE: {val_rmse:.4f} | "
                f"Val Pearson: {val_pr:.4f}"
            )
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_epoch = epoch

                torch.save(model.state_dict(), best_model_path)

                print(">>> Better model stocked")
        best_model = initialize_model("GAT")
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_model.to(device)

        test_rmse, test_pr = val(best_model, test_loader, device)

        print(
            f"[SPLIT {split_id}] "
            f"Best epoch: {best_epoch} | "
            f"Best Val RMSE: {best_rmse:.4f} | "
            f"Test RMSE: {test_rmse:.4f} | "
            f"Test Pearson: {test_pr:.4f}"
        )

        split_best_val_rmses.append(best_rmse)
        split_test_rmses.append(test_rmse)
        split_test_pearsons.append(test_pr)

    print("Training finished for all splits")
    print(f"Mean Best Val RMSE: {np.mean(split_best_val_rmses):.4f} std {np.std(split_best_val_rmses):.4f}")
    print(f"Mean Test RMSE: {np.mean(split_test_rmses):.4f} std  {np.std(split_test_rmses):.4f}")
    print(f"Mean Test Pearson: {np.mean(split_test_pearsons):.4f} std {np.std(split_test_pearsons):.4f}")
