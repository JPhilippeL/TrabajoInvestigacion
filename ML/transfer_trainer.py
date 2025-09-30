# transfer_trainer.py

import torch
import os
import gc
import logging
from sklearn.model_selection import train_test_split
from ML.model_trainer import create_model, train
from ML.data_processing import read_targets, load_data_from_sdf, create_dataloader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

CARPETA_MODELOS = "Modelos"

def transfer_train(
    pretrained_model_path,
    sdf_dir,
    target_file,
    transfer_mode="fine_tuning",  # 'feature_extraction' o 'fine_tuning'
    epochs=20,
    lr=1e-3,
    batch_size=32,
    valid_split=0.2,
    patience=0,
    model_name="transfer_model",
):
    # Validar transfer_mode
    if transfer_mode not in ["fine_tuning", "feature_extraction"]:
        raise ValueError("transfer_mode debe ser 'fine_tuning' o 'feature_extraction'")

    # Cargar checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(pretrained_model_path, map_location=device)

    model_type = checkpoint['model_type']
    input_dim = checkpoint['input_dim']
    edge_dim = checkpoint['edge_dim']
    hidden_dim = checkpoint.get('hidden_dim', 64)
    num_layers = checkpoint.get('num_layers', 3)

    # Cargar nuevo dataset
    target_dict = read_targets(target_file)
    target_name = os.path.splitext(os.path.basename(target_file))[0]
    data_list = load_data_from_sdf(sdf_dir, target_dict)

    if (valid_split > 0) and (valid_split < 1):
        train_data, val_data = train_test_split(data_list, test_size=valid_split, random_state=42)
        val_loader = create_dataloader(val_data, batch_size=batch_size)
    else:
        train_data = data_list
        val_loader = None
        
    train_loader = create_dataloader(train_data, batch_size=batch_size)

    # Crear modelo y cargar pesos
    model = create_model(model_type, input_dim, edge_dim, hidden_dim, num_layers)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # strict=False permite mismatch

    # Ajustar capa de salida si hay incompatibilidad
    if hasattr(model, 'lin'):
        out_features = model.lin.out_features
        if out_features != 1:  # asumimos que el nuevo target es un solo valor
            logging.info("Capa de salida incompatible, se reemplazará automáticamente")
            model.lin = torch.nn.Linear(hidden_dim, 1)
    elif hasattr(model, 'output'):
        out_features = model.output.out_features
        if out_features != 1:
            logging.info("Capa de salida incompatible, se reemplazará automáticamente")
            model.output = torch.nn.Linear(hidden_dim, 1)

    # Configurar Feature Extraction (congelar capas)
    if transfer_mode == "feature_extraction":
        logging.info("Modo Feature Extraction: capas previas congeladas")
        if hasattr(model, 'convs'):
            for param in model.convs.parameters():
                param.requires_grad = False
        if hasattr(model, 'node_encoder'):
            for param in model.node_encoder.parameters():
                param.requires_grad = False

    # Entrenar
    train(model, train_loader, device, epochs=epochs, lr=lr, val_loader=val_loader, patience=patience, model_name=model_name)

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
        'early_stopping_patience': patience,
        'transfer_mode': transfer_mode,
    }

    # Crear carpeta de modelos si no existe
    os.makedirs(CARPETA_MODELOS, exist_ok=True)
    # Guardar el modelo
    save_path = os.path.join(CARPETA_MODELOS, f"{model_name}.pt")
    torch.save(checkpoint_transfer, save_path)
    #logging.info(f"Modelo Transfer Learning guardado en: {save_path}")

    # Liberar memoria
    del model, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return save_path
