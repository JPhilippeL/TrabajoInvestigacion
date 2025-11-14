#model_trainer.py

import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import logging
import gc
import pandas as pd
import re

from torch_geometric.nn import GINConv, GINEConv, GATConv, global_add_pool, TransformerConv

from ML.data_processing import prepare_sdf_training_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)

from ui.utils import RESULTADOS_DIR, MODELOS_DIR, hybridization_types, periodic_elements, N_BOND_TYPES, ATOM_EMB_DIM, HYBRID_EMB_DIM, BOND_EMB_DIM, INPUT_DIM, EDGE_DIM
HEADS = 4  # Número de cabezas para GAT y GraphTransformer


class EmbeddingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # El embedding del atomo con el numero de atomos diferentes y la dimnension del vector que queremos
        self.atom_embedding = torch.nn.Embedding(len(periodic_elements), ATOM_EMB_DIM)
        # Lo mismo con lo demás
        self.hybrid_embedding = torch.nn.Embedding(len(hybridization_types), HYBRID_EMB_DIM)
        self.bond_embedding = torch.nn.Embedding(N_BOND_TYPES, BOND_EMB_DIM)

    def encode_nodes(self, x):
        symbol_idx = x[:, 0].long()
        hybrid_idx = x[:, 1].long()
        cont_features = x[:, 2:]  # [degree, numH, aromatic]

        symbol_emb = self.atom_embedding(symbol_idx)
        hybrid_emb = self.hybrid_embedding(hybrid_idx)
        return torch.cat([symbol_emb, hybrid_emb, cont_features], dim=1)

    def encode_edges(self, edge_attr):
        bond_idx = edge_attr[:, 0].long()
        bond_dist = edge_attr[:, 1].unsqueeze(1)
        bond_emb = self.bond_embedding(bond_idx)
        return torch.cat([bond_emb, bond_dist], dim=1)


# ----------------------
# Modelos GNN
# ----------------------    
class GINNet(torch.nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, fc_hidden_dim=128):
        super().__init__()

        self.encoder = EmbeddingEncoder()
        self.node_encoder = torch.nn.Linear(INPUT_DIM, hidden_dim)
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, fc_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr = None, batch = None):
        x = self.encoder.encode_nodes(x)
        x = self.node_encoder(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        out = self.fc(x)
        return out.view(-1)
        
    def get_embedding(self, x, edge_index, edge_attr=None, batch=None):
        x = self.node_encoder(x) if hasattr(self, 'node_encoder') else x
        for conv in self.convs:
            x = conv(x, edge_index) if edge_attr is None else conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        return x
    
class GINENet(torch.nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, fc_hidden_dim=128):
        super().__init__()

        self.encoder = EmbeddingEncoder()
        self.node_encoder = torch.nn.Linear(INPUT_DIM, hidden_dim)
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim = EDGE_DIM))

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, fc_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.encoder.encode_nodes(x)
        x = self.node_encoder(x)
        edge_attr = self.encoder.encode_edges(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        out = self.fc(x)
        return out.view(-1)
    
    def get_embedding(self, x, edge_index, edge_attr=None, batch=None):
        x = self.node_encoder(x) if hasattr(self, 'node_encoder') else x
        for conv in self.convs:
            x = conv(x, edge_index) if edge_attr is None else conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        return x
    
class GATNet(torch.nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, heads=4, fc_hidden_dim=128):
        super().__init__()

        self.encoder = EmbeddingEncoder()
        self.node_encoder = torch.nn.Linear(INPUT_DIM, hidden_dim)
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * heads
            conv = GATConv(in_channels, hidden_dim, heads=heads, concat=True)
            self.convs.append(conv)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * heads, fc_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.encoder.encode_nodes(x)
        x = self.node_encoder(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
        x = global_add_pool(x, batch)
        out = self.fc(x)
        return out.view(-1)
    
    def get_embedding(self, x, edge_index, edge_attr=None, batch=None):
        x = self.node_encoder(x) if hasattr(self, 'node_encoder') else x
        for conv in self.convs:
            x = conv(x, edge_index) if edge_attr is None else conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        return x
    
class EGATNet(torch.nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, heads=4, fc_hidden_dim=128):
        super().__init__()

        self.encoder = EmbeddingEncoder()
        self.node_encoder = torch.nn.Linear(INPUT_DIM, hidden_dim)
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * heads
            conv = GATConv(
                in_channels,
                hidden_dim, 
                heads=heads,
                edge_dim=EDGE_DIM, 
                concat=True)
            self.convs.append(conv)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * heads, fc_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.encoder.encode_nodes(x)
        x = self.node_encoder(x)
        edge_attr = self.encoder.encode_edges(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)
        x = global_add_pool(x, batch)
        out = self.fc(x)
        return out.view(-1)
    
    def get_embedding(self, x, edge_index, edge_attr=None, batch=None):
        x = self.node_encoder(x) if hasattr(self, 'node_encoder') else x
        for conv in self.convs:
            x = conv(x, edge_index) if edge_attr is None else conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        return x
    
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
    
    def get_embedding(self, x, edge_index, edge_attr=None, batch=None):
        x = self.node_encoder(x) if hasattr(self, 'node_encoder') else x
        for conv in self.convs:
            x = conv(x, edge_index) if edge_attr is None else conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        return x


# ----------------------
# Función para crear modelo según elección
# ----------------------

def create_model(model_name, input_dim, hidden_dim, num_layers, edge_dim):
    if model_name == "GIN":
        return GINNet(hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_name == "GINE":
        return GINENet(hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_name == "GAT":
        return GATNet(hidden_dim=hidden_dim, num_layers=num_layers, heads=HEADS)
    elif model_name == "GraphTransformer":
        return GraphTransformerNet(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, edge_dim=edge_dim, heads=HEADS)
    elif model_name == "EGAT":
        return EGATNet(hidden_dim=hidden_dim, num_layers=num_layers, heads=HEADS)
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

def save_model(
    model,
    model_name,
    input_dim,
    edge_dim,
    target_name,
    model_type,
    epochs,
    hidden_dim=64,
    num_layers=3,
    batch_size=32,
    lr=0.001,
    valid_split=0.2,
    patience=0,
    modelos_dir=MODELOS_DIR
):
    """
    Guarda un checkpoint de PyTorch con el estado del modelo y metadatos.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'input_dim': input_dim,
        'edge_dim': edge_dim,
        'epochs_trained': epochs,
        'target_name': target_name,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'batch_size': batch_size,
        'learning_rate': lr,
        'valid_split': valid_split,
        'early_stopping_patience': patience,
    }

    os.makedirs(modelos_dir, exist_ok=True)
    save_path = os.path.join(modelos_dir, f"{model_name}.pt")
    torch.save(checkpoint, save_path)

    # Limpiar memoria
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return save_path


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
    train_loader, val_loader, device, input_dim, edge_dim, targetname = prepare_sdf_training_data(
        sdf_dir, target_file, batch_size=batch_size, valid_split=valid_split
    )

    model = create_model(model_type, input_dim, hidden_dim=hidden_dim, num_layers=num_layers, edge_dim=edge_dim)

    train(model, train_loader, device, epochs=epochs, lr=lr, val_loader=val_loader, patience=patience, model_name=model_name)

    save_path = save_model(
        model=model,
        model_name=model_name,
        input_dim=input_dim,
        edge_dim=edge_dim,
        target_name=targetname,
        model_type=model_type,
        epochs=epochs,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        batch_size=batch_size,
        lr=lr,
        valid_split=valid_split,
        patience=patience
    )
    logging.info(f"Modelo guardado en: {save_path}")

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
    model_types = ["GIN", "GINE", "GAT", "EGAT", "GraphTransformer"]
    capas = [2, 3, 4, 5]
    nombreTarget = os.path.splitext(os.path.basename(target_file))[0]

    # Preparar datos
    train_loader, val_loader, device, input_dim, edge_dim, targetname = prepare_sdf_training_data(
        sdf_dir, target_file, batch_size=batch_size, valid_split=valid_split
    )

    for model_type in model_types:
        for num_layers in capas:
            model_name = f"{model_type}_{num_layers}capas_{nombreTarget}"
            logging.info(f"Entrenando modelo: {model_name}")
            # Crear modelo
            model = create_model(model_type, input_dim, hidden_dim=hidden_dim, num_layers=num_layers, edge_dim=edge_dim)
            # Entrenar
            train(model, train_loader, device, epochs=epochs, lr=lr, val_loader=val_loader, patience=patience, model_name=model_name)
            # Guardar
            save_path = save_model(
                model=model,
                model_name=model_name,
                input_dim=input_dim,
                edge_dim=edge_dim,
                target_name=targetname,
                model_type=model_type,
                epochs=epochs,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                batch_size=batch_size,
                lr=lr,
                valid_split=valid_split,
                patience=patience
            )
            logging.info(f"Modelo guardado en: {save_path}")