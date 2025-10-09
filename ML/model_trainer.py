#model_trainer.py

import sys
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv, GATConv, global_add_pool, TransformerConv
from sklearn.model_selection import train_test_split
from ML.data_processing import read_targets, load_data_from_sdf, create_dataloader
import matplotlib.pyplot as plt
import os
import logging
import gc
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)

from ui.utils import RESULTADOS_DIR, MODELOS_DIR
HEADS = 4  # Número de cabezas para GAT y GraphTransformer


# ----------------------
# Modelos GNN
# ----------------------

class GINNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, fc_hidden_dim=128):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, fc_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        out = self.fc(x)
        return out.squeeze()


class GINENet(torch.nn.Module):
    def __init__(self, input_dim, edge_dim=1, hidden_dim=64, num_layers=3, fc_hidden_dim=128):
        super().__init__()
        self.node_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, fc_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        return self.fc(x).view(-1)


class GATNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, heads=4, fc_hidden_dim=128):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim * heads
            conv = GATConv(in_channels, hidden_dim, heads=heads, concat=True)
            self.convs.append(conv)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * heads, fc_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
        x = global_add_pool(x, batch)
        out = self.fc(x)
        return out.squeeze()
    
class GraphTransformerNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, edge_dim=1, heads=4, fc_hidden_dim=128):
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

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * heads, fc_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        out = self.fc(x)
        return out.squeeze()


# ----------------------
# Función para crear modelo según elección
# ----------------------

def create_model(model_name, input_dim, hidden_dim, num_layers, edge_dim):
    if model_name == "GIN":
        return GINNet(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_name == "GINE":
        return GINENet(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, edge_dim=edge_dim)
    elif model_name == "GAT":
        return GATNet(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, heads=HEADS)
    elif model_name == "GraphTransformer":
        return GraphTransformerNet(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, edge_dim=edge_dim, heads=HEADS)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")


# ----------------------
# Función para entrenar modelo
# ----------------------

def train(model, train_loader, device, epochs=20, lr=0.001, val_loader=None, patience=0, model_name="model"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = epochs
    avg_train_loss_saved = None

    # --- Guardar histórico ---
    train_losses = []
    val_losses = []

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
        train_losses.append(avg_train_loss)

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
            val_losses.append(avg_val_loss)

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

    # --- Gráfica ---
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    if val_loader is not None:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')  # <- escala logarítmica
    plt.title(f'Training and Validation Loss - {model_name}')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    
    # Guardarla
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    model_results_dir = os.path.join(RESULTADOS_DIR, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    plt.savefig(os.path.join(model_results_dir, f"{model_name}_loss_curve.png"))
    plt.close()
    
    logging.info(f"Gráfico de pérdidas guardado en {os.path.join(model_results_dir, f'{model_name}_loss_curve.png')}")



def train_and_save_model(
    sdf_dir,
    target_file,
    model_type,
    epochs,
    model_name,
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
    if (valid_split > 0) and (valid_split < 1):
        train_data, val_data = train_test_split(data_list, test_size=valid_split, random_state=42)
        val_loader = create_dataloader(val_data, batch_size=batch_size)
    else:
        train_data = data_list
        val_loader = None
        
    train_loader = create_dataloader(train_data, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = data_list[0].x.shape[1]
    edge_dim = data_list[0].edge_attr.shape[1]

    model = create_model(model_type, input_dim, hidden_dim=hidden_dim, num_layers=num_layers, edge_dim=edge_dim)

    train(model, train_loader, device, epochs=epochs, lr=lr, val_loader=val_loader, patience=patience, model_name=model_name)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
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
    # Crear carpeta de modelos si no existe
    os.makedirs(MODELOS_DIR, exist_ok=True)
    # Guardar el modelo
    save_path = os.path.join(MODELOS_DIR, f"{model_name}.pt")

    torch.save(checkpoint, save_path)

    del model
    del train_loader
    del val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return save_path

# Entrenar y guardar modelos cambiandole las capas
def train_multiple_models(
    sdf_dir,
    target_file,
    epochs,
    batch_size=32,
    lr=0.001,
    valid_split=0.2,
    hidden_dim=64,
    patience=0
):
    model_types = ["GIN", "GINE", "GAT", "GraphTransformer"]
    capas = [2, 3, 4, 5]
    nombreTarget = os.path.splitext(os.path.basename(target_file))[0]
    saved_models = []
    for model_type in model_types:
        for num_layers in capas:
            model_name = f"{model_type}_{num_layers}capas_{nombreTarget}"
            logging.info(f"Entrenando modelo: {model_name}")
            save_path = train_and_save_model(
                sdf_dir=sdf_dir,
                target_file=target_file,
                model_type=model_type,
                epochs=epochs,
                model_name=model_name,
                batch_size=batch_size,
                lr=lr,
                valid_split=valid_split,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                patience=patience
            )
            saved_models.append(save_path)
            logging.info(f"Modelo guardado en: {save_path}")