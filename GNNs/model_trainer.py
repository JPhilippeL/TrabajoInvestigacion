#model_trainer.py

import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import logging
import gc
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.nn import GINConv, GINEConv, GATConv, global_add_pool, TransformerConv

from GNNs.data_processing import prepare_sdf_training_data, prepare_pt_training_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)

from ui.utils.constants import RESULTADOS_DIR, MODELOS_DIR, hybridization_types, periodic_elements, N_BOND_TYPES, ATOM_EMB_PR, HYBRID_EMB_PR, BOND_EMB_PR, OTHER_EDGE_FEATURES, OTHER_NODE_FEATURES
HEADS = 4  # Número de cabezas para GAT y GraphTransformer


class EmbeddingEncoder(torch.nn.Module):
    def __init__(self, atom_emb_dim, hybrid_emb_dim, bond_emb_dim):
        super().__init__()
        # El embedding del atomo con el numero de atomos diferentes y la dimnension del vector que queremos
        self.atom_embedding = torch.nn.Embedding(len(periodic_elements), atom_emb_dim)
        # Lo mismo con lo demás
        self.hybrid_embedding = torch.nn.Embedding(len(hybridization_types), hybrid_emb_dim)
        self.bond_embedding = torch.nn.Embedding(N_BOND_TYPES, bond_emb_dim)

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
    def __init__(self, input_dim, atom_emb_dim, hibrid_emb_dim, bond_emb_dim, hidden_dim=64, num_layers=3, fc_hidden_dim=128, dropout = 0.2):
        super().__init__()

        self.dropout = dropout  # Guardamos la probabilidad de dropout

        self.encoder = EmbeddingEncoder(atom_emb_dim, hibrid_emb_dim, bond_emb_dim,)
        self.node_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers):
            GNNsp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(GNNsp))

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, fc_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout), # Dropout antes de la capa final
            torch.nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr = None, batch = None):
        x = self.encoder.encode_nodes(x)
        x = self.node_encoder(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
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
    # Añadimos el argumento 'dropout' al init
    def __init__(self, input_dim, atom_emb_dim, hibrid_emb_dim, bond_emb_dim, edge_dim, hidden_dim=64, num_layers=3, fc_hidden_dim=128, dropout=0.2):
        super().__init__()
        
        self.dropout = dropout  # Guardamos la probabilidad de dropout

        self.encoder = EmbeddingEncoder(atom_emb_dim, hibrid_emb_dim, bond_emb_dim,)
        self.node_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            GNNsp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(GNNsp, edge_dim = edge_dim))

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, fc_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout), # Dropout antes de la capa final
            torch.nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.encoder.encode_nodes(x)
        x = self.node_encoder(x)
        edge_attr = self.encoder.encode_edges(edge_attr)
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            # Dropout después de cada bloque convolucional
            x = F.dropout(x, p=self.dropout, training=self.training)
            
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
    def __init__(self, input_dim, atom_emb_dim, hibrid_emb_dim, bond_emb_dim, hidden_dim=64, num_layers=3, heads=4, fc_hidden_dim=128, dropout = 0.2):
        super().__init__()

        self.dropout = dropout  # Guardamos la probabilidad de dropout

        self.encoder = EmbeddingEncoder(atom_emb_dim, hibrid_emb_dim, bond_emb_dim,)
        self.node_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * heads
            conv = GATConv(in_channels, hidden_dim, heads=heads, concat=True)
            self.convs.append(conv)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * heads, fc_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout), # Dropout antes de la capa final
            torch.nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.encoder.encode_nodes(x)
        x = self.node_encoder(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
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
    def __init__(self, input_dim, atom_emb_dim, hibrid_emb_dim, bond_emb_dim, edge_dim, hidden_dim=64, num_layers=3, heads=4, fc_hidden_dim=128, dropout = 0.2):
        super().__init__()

        self.dropout = dropout

        self.encoder = EmbeddingEncoder(atom_emb_dim, hibrid_emb_dim, bond_emb_dim,)
        self.node_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * heads
            conv = GATConv(
                in_channels,
                hidden_dim, 
                heads=heads,
                edge_dim=edge_dim, 
                concat=True)
            self.convs.append(conv)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * heads, fc_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout), # Dropout antes de la capa final
            torch.nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.encoder.encode_nodes(x)
        x = self.node_encoder(x)
        edge_attr = self.encoder.encode_edges(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
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
    def __init__(self, input_dim, atom_emb_dim, hibrid_emb_dim, bond_emb_dim, edge_dim, hidden_dim=64, num_layers=3, heads=4, fc_hidden_dim=128, dropout = 0.2):
        super().__init__()

        self.dropout = dropout

        self.encoder = EmbeddingEncoder(atom_emb_dim, hibrid_emb_dim, bond_emb_dim,)
        self.node_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * heads
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
            torch.nn.Dropout(p=dropout), # Dropout antes de la capa final
            torch.nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.encoder.encode_nodes(x)
        x = self.node_encoder(x)
        edge_attr = self.encoder.encode_edges(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
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


# ----------------------
# Función para crear modelo según elección
# ----------------------

def create_model(model_name, input_dim, atom_emb_dim, hibrid_emb_dim, bond_emb_dim, hidden_dim, num_layers, edge_dim):
    if model_name == "GIN":
        return GINNet(input_dim, atom_emb_dim, hibrid_emb_dim, bond_emb_dim, hidden_dim, num_layers)
    elif model_name == "GINE":
        return GINENet(input_dim, atom_emb_dim, hibrid_emb_dim, bond_emb_dim, edge_dim, hidden_dim, num_layers)
    elif model_name == "GAT":
        return GATNet(input_dim,atom_emb_dim, hibrid_emb_dim, bond_emb_dim, hidden_dim, num_layers, HEADS)
    elif model_name == "GraphTransformer":
        return GraphTransformerNet(input_dim, atom_emb_dim, hibrid_emb_dim, bond_emb_dim, edge_dim, hidden_dim, num_layers, HEADS)
    elif model_name == "EGAT":
        return EGATNet(input_dim, atom_emb_dim, hibrid_emb_dim, bond_emb_dim, edge_dim, hidden_dim, num_layers, HEADS)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")


# ----------------------
# Función para entrenar modelo
# ----------------------

def train(model, train_loader, device, epochs=20, lr=0.001, val_loader=None, patience=0, model_name="model"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # --- CAMBIO 1: Scheduler ---
    # Si la valid loss no mejora en 'patience_scheduler' épocas, reducimos el LR a la mitad (factor 0.5)
    patience_scheduler = max(10, patience // 4) if patience > 0 else 15
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_scheduler)
    
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = epochs
    avg_train_loss_saved = None

    # --- Guardar histórico ---
    train_losses = []
    val_losses = []
    lrs = [] # Para graficar el LR si quisieras

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

            # --- CAMBIO 2: Step del Scheduler ---
            # Le decimos al scheduler cómo nos fue en validación
            scheduler.step(avg_val_loss)
            
            # Guardar el LR actual para debug
            current_lr = optimizer.param_groups[0]['lr']

            # Guardar el mejor modelo si la validación mejora
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model_tmp.pt")
                best_epoch = epoch
                avg_train_loss_saved = avg_train_loss
            else:
                if patience > 0:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"Early stopping en epoch {epoch}")
                        break

        del batch
        if avg_val_loss is not None:
            logging.info(f"Epoch {epoch:03d} | LR: {current_lr:.6f} | Train MSE: {avg_train_loss:.4f} | Validation MSE: {avg_val_loss:.4f}")
        else:
            logging.info(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f}")

    # Restaurar siempre el mejor modelo antes de salir
    if os.path.exists("best_model_tmp.pt"):
        model.load_state_dict(torch.load("best_model_tmp.pt"))
        os.remove("best_model_tmp.pt")
        logging.info(f"Mejor modelo guardado en epoch {best_epoch} | Train MSE: {avg_train_loss_saved:.4f} | Validation MSE: {best_val_loss:.4f}")

    # --- Gráfica ---
    plt.figure(figsize=(10, 6)) # Hacemos la figura un poco más grande
    
    # Eje X basado en los datos reales obtenidos
    epochs_range = range(1, len(train_losses) + 1)
    
    plt.plot(epochs_range, train_losses, label='Train Loss')
    if val_loader is not None:
        plt.plot(epochs_range, val_losses, label='Validation Loss')

    # --- NUEVO: Marcar Early Stopping y Best Epoch ---
    
    # 1. Marca del Mejor Modelo (Línea Verde)
    # Solo la dibujamos si hubo validación y tenemos un best_epoch guardado
    if val_loader is not None and best_epoch:
        plt.axvline(x=best_epoch, color='green', linestyle=':', label=f'Best Epoch ({best_epoch})')

    # 2. Marca de Early Stopping (Línea Roja)
    # Si la longitud del histórico es menor que los epochs totales, hubo early stopping
    actual_epochs = len(train_losses)
    if actual_epochs < epochs:
        plt.axvline(x=actual_epochs, color='red', linestyle='--', label=f'Early Stopping ({actual_epochs})')
        # Opcional: Añadir texto en la gráfica
        plt.text(actual_epochs, plt.ylim()[1]*0.9, ' Stop', color='red', ha='right')

    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
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
    modelos_dir=MODELOS_DIR,
    atom_emb_dim = ATOM_EMB_PR,
    hibrid_emb_dim = HYBRID_EMB_PR,
    bond_emb_dim = BOND_EMB_PR
):
    """
    Guarda un checkpoint de PyTorch con el estado del modelo y metadatos.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'input_dim': input_dim,
        "atom_emb_dim" : atom_emb_dim,
        "hibrid_emb_dim" : hibrid_emb_dim,
        "bond_emb_dim" : bond_emb_dim,
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
    model_name,  # Este es el nombre "sugerido" por el usuario
    batch_size=32,
    lr=0.001,
    valid_split=0.2,
    hidden_dim=64,
    num_layers=3,
    patience=0,
    atom_emb_dim = ATOM_EMB_PR,
    hibrid_emb_dim = HYBRID_EMB_PR,
    bond_emb_dim = BOND_EMB_PR
):
    # 1. Calcular dimensiones y cargar datos
    train_loader, val_loader, device, targetname = prepare_sdf_training_data(
        sdf_dir, target_file, batch_size=batch_size, valid_split=valid_split
    )
    
    calc_atom_emb_dim = calc_dim(len(periodic_elements) * atom_emb_dim)
    calc_hibrid_emb_dim = calc_dim(len(hybridization_types) * hibrid_emb_dim)
    calc_bond_emb_dim = calc_dim(N_BOND_TYPES * bond_emb_dim)

    input_dim = calc_atom_emb_dim + calc_hibrid_emb_dim + OTHER_NODE_FEATURES
    edge_dim = calc_bond_emb_dim + OTHER_EDGE_FEATURES

    # --- NUEVO: CALCULAR NOMBRE ÚNICO AQUÍ ---
    # Buscamos qué nombre está libre en la carpeta de modelos.
    # Así usamos ESE MISMO nombre para la gráfica y para el guardado.
    final_model_name = get_unique_name(model_name, MODELOS_DIR, extension=".pt")
    
    logging.info(f"Iniciando entrenamiento para: {final_model_name}")

    # 2. Crear modelo
    model = create_model(
        model_type,
        input_dim,
        calc_atom_emb_dim,
        calc_hibrid_emb_dim, 
        calc_bond_emb_dim, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        edge_dim=edge_dim)

    # 3. Entrenar (Pasamos final_model_name para que la gráfica no se sobrescriba)
    train(model, train_loader, device, epochs=epochs, lr=lr, val_loader=val_loader, patience=patience, model_name=final_model_name)

    # 4. Guardar (Pasamos final_model_name para que coincida con la gráfica)
    save_path = save_model(
        model=model,
        model_name=final_model_name, # <--- Usamos el nombre único
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
        patience=patience,
        atom_emb_dim = atom_emb_dim,
        hibrid_emb_dim = hibrid_emb_dim,
        bond_emb_dim = bond_emb_dim
    )
    logging.info(f"Modelo y gráficas guardados bajo el ID: {final_model_name}")

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
    patience=0,
    atom_emb_dim = ATOM_EMB_PR,
    hibrid_emb_dim = HYBRID_EMB_PR,
    bond_emb_dim = BOND_EMB_PR
):
    model_types = ["GIN", "GINE", "GAT", "EGAT", "GraphTransformer"]
    capas = [2, 3, 4, 5]
    nombreTarget = os.path.splitext(os.path.basename(target_file))[0]

    # Preparar datos
    train_loader, val_loader, device, targetname = prepare_sdf_training_data(
        sdf_dir, target_file, batch_size=batch_size, valid_split=valid_split
    )

    calc_atom_emb_dim = calc_dim(len(periodic_elements) * atom_emb_dim)
    calc_hibrid_emb_dim = calc_dim(len(hybridization_types) * hibrid_emb_dim)
    calc_bond_emb_dim = calc_dim(N_BOND_TYPES * bond_emb_dim)

    # Son porcentajes por los que se multiplican las dimensiones reales, 
    # de esta manera el usuario elige si quiere desde 1 dimension sola hasta el 100%
    input_dim = calc_atom_emb_dim + calc_hibrid_emb_dim + OTHER_NODE_FEATURES
    edge_dim = calc_bond_emb_dim + OTHER_EDGE_FEATURES

    for model_type in model_types:
        for num_layers in capas:
            model_name = f"{model_type}_{num_layers}capas_{nombreTarget}"
            logging.info(f"Entrenando modelo: {model_name}")
            # Crear modelo
            model = create_model(
                model_type,
                input_dim,
                calc_atom_emb_dim,
                calc_hibrid_emb_dim, 
                calc_bond_emb_dim, 
                hidden_dim=hidden_dim, 
                num_layers=num_layers, 
                edge_dim=edge_dim)
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
                patience=patience,
                atom_emb_dim = atom_emb_dim,
                hibrid_emb_dim = hibrid_emb_dim,
                bond_emb_dim = bond_emb_dim
            )
            logging.info(f"Modelo guardado en: {save_path}")

def train_and_save_model_from_pt(
    pt_file,
    model_type,
    epochs,
    model_name,  # Este es el nombre "sugerido" por el usuario
    batch_size=32,
    lr=0.001,
    valid_split=0.2,
    hidden_dim=64,
    num_layers=3,
    patience=0,
    atom_emb_dim = ATOM_EMB_PR,
    hibrid_emb_dim = HYBRID_EMB_PR,
    bond_emb_dim = BOND_EMB_PR
):
    # 1. Calcular dimensiones y cargar datos
    train_loader, val_loader, device, targetname = prepare_pt_training_data(
        pt_file, batch_size=batch_size, valid_split=valid_split
    )

    # Cambia esto temporalmente
    device = torch.device("cpu")
    
    calc_atom_emb_dim = calc_dim(len(periodic_elements) * atom_emb_dim)
    calc_hibrid_emb_dim = calc_dim(len(hybridization_types) * hibrid_emb_dim)
    calc_bond_emb_dim = calc_dim(N_BOND_TYPES * bond_emb_dim)

    input_dim = calc_atom_emb_dim + calc_hibrid_emb_dim + OTHER_NODE_FEATURES
    edge_dim = calc_bond_emb_dim + OTHER_EDGE_FEATURES

    # --- NUEVO: CALCULAR NOMBRE ÚNICO AQUÍ ---
    # Buscamos qué nombre está libre en la carpeta de modelos.
    # Así usamos ESE MISMO nombre para la gráfica y para el guardado.
    final_model_name = get_unique_name(model_name, MODELOS_DIR, extension=".pt")
    
    logging.info(f"Iniciando entrenamiento para: {final_model_name}")

    # 2. Crear modelo
    model = create_model(
        model_type,
        input_dim,
        calc_atom_emb_dim,
        calc_hibrid_emb_dim, 
        calc_bond_emb_dim, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        edge_dim=edge_dim)

    # 3. Entrenar (Pasamos final_model_name para que la gráfica no se sobrescriba)
    train(model, train_loader, device, epochs=epochs, lr=lr, val_loader=val_loader, patience=patience, model_name=final_model_name)

    # 4. Guardar (Pasamos final_model_name para que coincida con la gráfica)
    save_path = save_model(
        model=model,
        model_name=final_model_name, # <--- Usamos el nombre único
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
        patience=patience,
        atom_emb_dim = atom_emb_dim,
        hibrid_emb_dim = hibrid_emb_dim,
        bond_emb_dim = bond_emb_dim
    )
    logging.info(f"Modelo y gráficas guardados bajo el ID: {final_model_name}")

    return save_path

def calc_dim(x):
    return max(1, math.ceil(x))

def get_unique_name(base_name, directory, extension=".pt"):
    """
    Busca un nombre libre. Si 'base_name.pt' existe, prueba 'base_name_1.pt', etc.
    Devuelve el nombre base único (sin extensión) para usarlo en todo el proceso.
    """
    os.makedirs(directory, exist_ok=True)
    
    counter = 1
    # Empezamos asumiendo que el nombre base es el bueno
    unique_name = base_name
    
    # Mientras exista el archivo en la carpeta de MODELOS, seguimos buscando
    while os.path.exists(os.path.join(directory, f"{unique_name}{extension}")):
        unique_name = f"{base_name}_{counter}"
        counter += 1
        
    return unique_name