import optuna
import torch
import numpy as np
import logging
from ML.model_trainer import (
    create_model, train, save_model, prepare_sdf_training_data,
    calc_dim, periodic_elements, hybridization_types, N_BOND_TYPES,
    OTHER_NODE_FEATURES, OTHER_EDGE_FEATURES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def objective(trial, model_type, sdf_dir, target_file, epochs=20):
    """
    Función objetivo para Optuna: entrena un modelo con los hiperparámetros sugeridos
    y devuelve el RMSE en el conjunto de validación.
    """
    # Espacio de hiperparámetros
    num_layers = trial.suggest_int("num_layers", 2, 5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    atom_emb_dim = trial.suggest_int("atom_emb_dim", 1, 16)
    hibrid_emb_dim = trial.suggest_int("hibrid_emb_dim", 1, 16)
    bond_emb_dim = trial.suggest_int("bond_emb_dim", 1, 16)
    
    # Preparar datos
    train_loader, val_loader, device, targetname = prepare_sdf_training_data(
        sdf_dir, target_file, batch_size=32, valid_split=0.2
    )

    # Calcular dimensiones de embeddings
    calc_atom_emb_dim = calc_dim(len(periodic_elements) * atom_emb_dim)
    calc_hibrid_emb_dim = calc_dim(len(hybridization_types) * hibrid_emb_dim)
    calc_bond_emb_dim = calc_dim(N_BOND_TYPES * bond_emb_dim)
    input_dim = calc_atom_emb_dim + calc_hibrid_emb_dim + OTHER_NODE_FEATURES
    edge_dim = calc_bond_emb_dim + OTHER_EDGE_FEATURES

    # Crear modelo
    model = create_model(
        model_type,
        input_dim,
        calc_atom_emb_dim,
        calc_hibrid_emb_dim,
        calc_bond_emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        edge_dim=edge_dim
    )

    # Entrenar modelo
    train(model, train_loader, device, epochs=epochs, lr=lr, val_loader=val_loader, patience=5, model_name=f"trial_{model_type}")

    # Evaluar RMSE en validación
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y_pred.extend(out.cpu().numpy())
            y_true.extend(batch.y.cpu().numpy())
    rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

    return rmse

def buscar_mejor_modelo_por_arquitectura(sdf_dir, target_file, epochs=20, n_trials=30, modelos_dir="models"):
    """
    Ejecuta Optuna para cada arquitectura y guarda el mejor modelo entrenado.
    """
    from os import makedirs
    makedirs(modelos_dir, exist_ok=True)

    mejores_modelos = {}

    for model_type in ["GIN", "GINE", "GAT", "EGAT", "GraphTransformer"]:
        logger.info(f"Buscando mejores hiperparámetros para {model_type}...")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, model_type, sdf_dir, target_file, epochs=epochs), n_trials=n_trials)

        best_params = study.best_params
        logger.info(f"Mejores parámetros para {model_type}: {best_params}")

        # Entrenar de nuevo el mejor modelo y guardarlo
        num_layers = best_params["num_layers"]
        hidden_dim = best_params["hidden_dim"]
        lr = best_params["lr"]
        atom_emb_dim = best_params["atom_emb_dim"]
        hibrid_emb_dim = best_params["hibrid_emb_dim"]
        bond_emb_dim = best_params["bond_emb_dim"]

        # Preparar datos
        train_loader, val_loader, device, targetname = prepare_sdf_training_data(
            sdf_dir, target_file, batch_size=32, valid_split=0.2
        )

        calc_atom_emb_dim = calc_dim(len(periodic_elements) * atom_emb_dim)
        calc_hibrid_emb_dim = calc_dim(len(hybridization_types) * hibrid_emb_dim)
        calc_bond_emb_dim = calc_dim(N_BOND_TYPES * bond_emb_dim)
        input_dim = calc_atom_emb_dim + calc_hibrid_emb_dim + OTHER_NODE_FEATURES
        edge_dim = calc_bond_emb_dim + OTHER_EDGE_FEATURES

        model = create_model(
            model_type,
            input_dim,
            calc_atom_emb_dim,
            calc_hibrid_emb_dim,
            calc_bond_emb_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            edge_dim=edge_dim
        )

        # Entrenar con mejores parámetros
        train(model, train_loader, device, epochs=epochs, lr=lr, val_loader=val_loader, patience=5, model_name=f"best_{model_type}")

        # Guardar el modelo
        save_model(
            model=model,
            model_name=f"best_{model_type}",
            input_dim=input_dim,
            edge_dim=edge_dim,
            target_name=targetname,
            model_type=model_type,
            epochs=epochs,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_size=32,
            lr=lr,
            valid_split=0.2,
            patience=5
        )

        mejores_modelos[model_type] = f"{modelos_dir}/best_{model_type}.pt"

    return mejores_modelos

# Uso:
# mejores = buscar_mejor_modelo_por_arquitectura("ruta_sdf", "targets.txt", epochs=20, n_trials=30)
# print(mejores)
