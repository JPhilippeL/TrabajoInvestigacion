# transfer_trainer_flexible.py

import torch
import os
import gc
import logging
from sklearn.model_selection import train_test_split
from ML.model_trainer import create_model, train
from ML.data_processing import read_targets, load_data_from_sdf, create_dataloader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def transfer_train_flexible(
    pretrained_model_path,
    sdf_dir,
    target_file,
    transfer_mode="fine_tuning",  # 'feature_extraction' o 'fine_tuning'
    epochs=20,
    lr=1e-3,
    batch_size=32,
    valid_split=0.2,
    save_path="transfer_model.pt"
):
    """
    Reentrena un modelo preentrenado (interno o externo) usando Transfer Learning.
    Detecta incompatibilidades de dimensiones y ajusta la capa de salida automáticamente.
    """
    # Validar transfer_mode
    if transfer_mode not in ["fine_tuning", "feature_extraction"]:
        raise ValueError("transfer_mode debe ser 'fine_tuning' o 'feature_extraction'")

    # Cargar checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(pretrained_model_path, map_location=device)

    model_type = checkpoint.get('model_type', None)
    hidden_dim = checkpoint.get('hidden_dim', 64)
    num_layers = checkpoint.get('num_layers', 3)

    if model_type is None:
        raise ValueError("El checkpoint no contiene 'model_type'. No se puede reconstruir el modelo.")

    # Cargar nuevo dataset
    target_dict = read_targets(target_file)
    target_name = os.path.splitext(os.path.basename(target_file))[0]
    data_list = load_data_from_sdf(sdf_dir, target_dict)

    # Detectar dimensiones del dataset
    try:
        input_dim = data_list[0].x.shape[1]
        edge_dim = data_list[0].edge_attr.shape[1]
    except Exception as e:
        raise ValueError(f"No se pudieron extraer dimensiones del dataset: {e}")

    # Validación de compatibilidad mínima
    checkpoint_input_dim = checkpoint.get('input_dim', None)
    checkpoint_edge_dim = checkpoint.get('edge_dim', None)

    if checkpoint_input_dim is not None and checkpoint_input_dim != input_dim:
        logging.warning(f"Input_dim del checkpoint ({checkpoint_input_dim}) no coincide con el dataset ({input_dim})")
    if checkpoint_edge_dim is not None and checkpoint_edge_dim != edge_dim:
        logging.warning(f"Edge_dim del checkpoint ({checkpoint_edge_dim}) no coincide con el dataset ({edge_dim})")

    # Crear modelo
    model = create_model(model_type, input_dim=input_dim, edge_dim=edge_dim,
                         hidden_dim=hidden_dim, num_layers=num_layers)

    # Cargar pesos del checkpoint (flexible)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Ajustar capa de salida si es incompatible
    replaced_output = False
    if hasattr(model, 'lin'):
        if model.lin.out_features != 1:
            logging.info("Capa de salida incompatible detectada en 'lin'. Se reemplazará automáticamente.")
            model.lin = torch.nn.Linear(hidden_dim, 1)
            replaced_output = True
    elif hasattr(model, 'output'):
        if model.output.out_features != 1:
            logging.info("Capa de salida incompatible detectada en 'output'. Se reemplazará automáticamente.")
            model.output = torch.nn.Linear(hidden_dim, 1)
            replaced_output = True

    # Feature Extraction: congelar capas previas
    if transfer_mode == "feature_extraction":
        logging.info("Modo Feature Extraction activado: capas previas congeladas")
        if hasattr(model, 'convs'):
            for param in model.convs.parameters():
                param.requires_grad = False
        if hasattr(model, 'node_encoder'):
            for param in model.node_encoder.parameters():
                param.requires_grad = False

    # Preparar dataloaders
    train_data, val_data = train_test_split(data_list, test_size=valid_split, random_state=42)
    train_loader = create_dataloader(train_data, batch_size=batch_size)
    val_loader = create_dataloader(val_data, batch_size=batch_size)

    # Entrenamiento
    train_flexible(model, train_loader, device, epochs=epochs, lr=lr, val_loader=val_loader, patience=0)

    # Guardar checkpoint actualizado
    checkpoint_transfer = {
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
        'transfer_mode': transfer_mode,
        'replaced_output_layer': replaced_output,
        'external_model': checkpoint_input_dim != input_dim or checkpoint_edge_dim != edge_dim
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint_transfer, save_path)
    logging.info(f"Modelo Transfer Learning guardado en: {save_path}")

    # Liberar memoria
    del model, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return save_path


def train_flexible(
    model,
    train_loader,
    device,
    epochs=20,
    lr=1e-3,
    val_loader=None,
    patience=0,
    loss_fn=None
):
    """
    Función de entrenamiento flexible para Transfer Learning.
    Soporta modelos externos con edge_attr opcional y output variable.
    """

    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = loss_fn if loss_fn is not None else torch.nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = epochs
    avg_train_loss_saved = None
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Manejar edge_attr opcional
            try:
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            except TypeError:
                # Modelo no espera edge_attr
                out = model(batch.x, batch.edge_index, batch.batch)

            # Aplanar output
            out = out.view(-1)

            # Calcular loss
            y_true = batch.y.view(-1)
            loss = criterion(out, y_true)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validación
        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    try:
                        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    except TypeError:
                        out = model(batch.x, batch.edge_index, batch.batch)
                    out = out.view(-1)
                    y_true = batch.y.view(-1)
                    val_loss += criterion(out, y_true).item() * batch.num_graphs
            avg_val_loss = val_loss / len(val_loader.dataset)

            # Guardar mejor modelo
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                avg_train_loss_saved = avg_train_loss
            else:
                if patience > 0:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"Early stopping en epoch {epoch}")
                        break

        # Logging
        if avg_val_loss is not None:
            logging.info(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        else:
            logging.info(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f}")

        del batch

    # Restaurar mejor modelo
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.cuda.empty_cache()
    gc.collect()
    return model