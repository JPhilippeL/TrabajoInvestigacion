#model_trainer.py

import sys
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv, GATConv, global_add_pool, TransformerConv
from sklearn.model_selection import train_test_split
from ML.data_processing import read_targets, load_data_from_sdf, create_dataloader
import os
import logging
import gc
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)


# ----------------------
# Modelos GNN
# ----------------------

class GINNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINConv(mlp)
            self.convs.append(conv)
        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        out = self.lin(x)
        return out.squeeze()


class GINENet(torch.nn.Module):
    def __init__(self, input_dim, edge_dim=1, hidden_dim=64, num_layers=3):
        super().__init__()
        self.node_encoder = torch.nn.Linear(input_dim, hidden_dim)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINEConv(mlp, edge_dim=edge_dim)
            self.convs.append(conv)

        self.readout = global_add_pool
        self.output = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = self.readout(x, batch)
        return self.output(x).view(-1)


class GATNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, heads=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim * heads
            conv = GATConv(in_channels, hidden_dim, heads=heads, concat=True)
            self.convs.append(conv)
        self.lin = torch.nn.Linear(hidden_dim * heads, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
        x = global_add_pool(x, batch)
        out = self.lin(x)
        return out.squeeze()
    
class GraphTransformerNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, heads=4, edge_dim=1):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim * heads
            conv = TransformerConv(
                in_channels=in_channels,
                out_channels=hidden_dim,
                heads=heads,
                edge_dim=edge_dim,
                concat=True
            )
            self.convs.append(conv)
        self.lin = torch.nn.Linear(hidden_dim * heads, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        out = self.lin(x)
        return out.squeeze()


# ----------------------
# Función para crear modelo según elección
# ----------------------

def create_model(model_name, input_dim, edge_dim=1, hidden_dim=64, num_layers=3):
    if model_name == "GIN":
        return GINNet(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_name == "GINE":
        return GINENet(input_dim=input_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_name == "GAT":
        return GATNet(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_name == "GraphTransformer":
        return GraphTransformerNet(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")


# ----------------------
# Función para entrenar modelo
# ----------------------

def train(model, train_loader, device, epochs=20, lr=0.001, val_loader=None, patience=0):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = epochs
    avg_train_loss_saved = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        avg_train_loss = total_loss / len(train_loader.dataset)

        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    loss = criterion(out, batch.y)
                    val_loss += loss.item() * batch.num_graphs
            avg_val_loss = val_loss / len(val_loader.dataset)

            # Guardar el mejor modelo si la validación mejora
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model_tmp.pt")
                best_epoch = epoch
                avg_train_loss_saved = avg_train_loss
            else:
                if patience > 0:
                    # Early stopping
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"Early stopping en epoch {epoch}")
                        break

        del batch
        if avg_val_loss is not None:
            logging.info(f"Epoch {epoch:03d} | Train MSE: {avg_train_loss:.4f} | Validation MSE: {avg_val_loss:.4f}")
        else:
            logging.info(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f}")

    # Restaurar siempre el mejor modelo antes de salir
    if os.path.exists("best_model_tmp.pt"):
        model.load_state_dict(torch.load("best_model_tmp.pt"))
        os.remove("best_model_tmp.pt")
        logging.info(f"Mejor modelo guardado en epoch {best_epoch} | Train MSE: {avg_train_loss_saved:.4f} | Validation MSE: {best_val_loss:.4f}")




def train_and_save_model(
    sdf_dir,
    target_file,
    modelo_nombre,
    epochs,
    save_path,
    batch_size=32,
    lr=0.001,
    valid_split=0.2,
    hidden_dim=64,
    num_layers=3,
    patience=0
):
    target_dict = read_targets(target_file)
    targetname = os.path.splitext(os.path.basename(target_file))[0]
    data_list = load_data_from_sdf(sdf_dir, target_dict)

    train_data, val_data = train_test_split(data_list, test_size=valid_split, random_state=42)
    train_loader = create_dataloader(train_data, batch_size=batch_size)
    val_loader = create_dataloader(val_data, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = data_list[0].x.shape[1]
    edge_dim = data_list[0].edge_attr.shape[1]
    
    model = create_model(modelo_nombre, input_dim, edge_dim, hidden_dim=hidden_dim, num_layers=num_layers)

    train(model, train_loader, device, epochs=epochs, lr=lr, val_loader=val_loader, patience=patience)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': modelo_nombre,
        'input_dim': input_dim,
        'edge_dim': edge_dim,
        'epochs_trained': epochs,
        'target_name': targetname,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'batch_size': batch_size,
        'learning_rate': lr,
        'valid_split': valid_split,
        'early_stopping_patience': patience,
    }


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)

    del model
    del train_loader
    del val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return save_path

