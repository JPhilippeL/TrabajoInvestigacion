#model_tester.py

import torch
from rdkit import Chem
from torch_geometric.data import Data
from ML.model_trainer import create_model, calc_dim
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from ML.data_processing import read_targets, load_data_from_sdf, mol_to_graph_data
import logging
import sys
from ui.utils import RESULTADOS_DIR, hybridization_types, periodic_elements, N_BOND_TYPES, OTHER_EDGE_FEATURES, OTHER_NODE_FEATURES
import csv
from scipy.stats import pearsonr

# Logger por módulo (no tocar basicConfig aquí)
logger = logging.getLogger(__name__)

# Reducir logs de librerías ruidosas
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)

def cargar_modelo(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Recuperar metadatos
    model_type = checkpoint['model_type']
    input_dim = checkpoint['input_dim']
    edge_dim = checkpoint['edge_dim']
    hidden_dim = checkpoint.get('hidden_dim', 64)
    num_layers = checkpoint.get('num_layers', 3)
    target_name = checkpoint.get('target_name', 'target')
    atom_emb_dim = checkpoint.get('atom_emb_dim')
    hibrid_emb_dim = checkpoint.get('hibrid_emb_dim')
    bond_emb_dim = checkpoint.get('bond_emb_dim')

    calc_atom_emb_dim = calc_dim(len(periodic_elements) * atom_emb_dim)
    calc_hibrid_emb_dim = calc_dim(len(hybridization_types) * hibrid_emb_dim)
    calc_bond_emb_dim = calc_dim(N_BOND_TYPES * bond_emb_dim)

    # Son porcentajes por los que se multiplican las dimensiones reales, 
    # de esta manera el usuario elige si quiere desde 1 dimension sola hasta el 100%
    input_dim = calc_atom_emb_dim + calc_hibrid_emb_dim + OTHER_NODE_FEATURES
    edge_dim = calc_bond_emb_dim + OTHER_EDGE_FEATURES

    # Crear modelo con los parámetros guardados
    model = create_model(
        model_type,
        input_dim,
        calc_atom_emb_dim,
        calc_hibrid_emb_dim, 
        calc_bond_emb_dim, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        edge_dim=edge_dim)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, device, target_name

def cargar_y_predecir(checkpoint_path, sdf_path):
    model, device, target_name = cargar_modelo(checkpoint_path)

    # Leer molécula del SDF
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = suppl[0]
    if mol is None:
        raise ValueError(f"No se pudo leer la molécula de {sdf_path}")

    # Convertir a PyG Data (embedding)
    data = mol_to_graph_data(mol)
    data = data.to(str(device))

    # Predecir
    with torch.no_grad():
        num_nodes = data.num_nodes if data.num_nodes is not None else (data.x.shape[0] if data.x is not None else 1)
        batch_device = data.x.device if data.x is not None else device
        batch = torch.zeros(num_nodes, dtype=torch.long, device=batch_device)  # todos nodos del mismo grafo
        out = model(data.x, data.edge_index, data.edge_attr, batch)
        pred = out.squeeze().item()

    return pred, target_name

# Funcion para predecir una molecula teniendo el modelo ya cargado
# Suponemos que la molecula ya esta en formato Data de PyG
def predecir_molecula(model, data, device):
    data = data.to(str(device))
    
    with torch.no_grad():
        num_nodes = data.num_nodes if data.num_nodes is not None else (data.x.shape[0] if data.x is not None else 1)
        batch_device = data.x.device if data.x is not None else device
        batch = torch.zeros(num_nodes, dtype=torch.long, device=batch_device)  # todos nodos del mismo grafo
        out = model(data.x, data.edge_index, data.edge_attr, batch)
        pred = out.squeeze().item()
    return pred

def test_model_on_directory(checkpoint_path, sdf_dir, targets_file):
    try:
        model, device, target_name = cargar_modelo(checkpoint_path)

        # Crear carpeta de resultados si no existe
        os.makedirs(RESULTADOS_DIR, exist_ok=True)

        # Crear carpeta específica para este modelo
        model_filename = os.path.basename(checkpoint_path)
        model_name_no_ext = os.path.splitext(model_filename)[0]

        model_results_dir = os.path.join(RESULTADOS_DIR, model_name_no_ext)
        os.makedirs(model_results_dir, exist_ok=True)

        # Leer datos
        target_dict = read_targets(targets_file)
        data_list = load_data_from_sdf(sdf_dir, target_dict)

        y_true, y_pred, filenames = [], [], []

        for data in data_list:
            data = data.to(device)
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

            with torch.no_grad():
                out = model(data.x, data.edge_index, data.edge_attr, batch)
                pred = out.squeeze().item()

            y_pred.append(pred)
            y_true.append(data.y.item())
            filenames.append(data.name if hasattr(data, 'name') else 'unknown')

        # Nombre base de archivos
        folder_name = os.path.basename(sdf_dir.rstrip(os.sep))

        # Archivo de predicciones
        output_predictions_path = os.path.join(
            model_results_dir,
            f"predicciones_{model_name_no_ext}_{folder_name}.txt"
        )

        with open(output_predictions_path, 'w') as f:
            for fname, pred in zip(filenames, y_pred):
                f.write(f"{fname} {pred:.4f}\n")

        # RMSE
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        logger.info(f"RMSE: {rmse:.4f}")

        # Pearson coefficient
        if len(y_true) > 1:  # necesario para scipy
            pearson_r, _ = pearsonr(y_true, y_pred)
            logger.info(f"Pearson coefficient: {pearson_r:.4f}")
        else:
            pearson_r = float("nan")
            logger.info("Pearson coefficient: No se puede calcular con un solo punto.")

        # Scatter plot
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.7)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
        plt.xlabel("Valor real")
        plt.ylabel("Predicción")
        plt.title(f"Scatter Plot - {model_name_no_ext} - {folder_name}")
        plt.grid(True)
        plt.tight_layout()

        # Guardar imagen
        plot_filename = os.path.join(
            model_results_dir,
            f"scatter_plot_{model_name_no_ext}_{folder_name}.png"
        )
        plt.savefig(plot_filename)
        plt.close()

        logger.info(f"Scatter plot guardado en: {plot_filename}")
        logger.info(f"Predicciones guardadas en: {output_predictions_path}")

        # Hacemos return del path del plot
        return plot_filename

    except Exception as e:
        raise ValueError(e)


def obtener_info_checkpoint(model_path):
    """
    Carga un checkpoint y devuelve un resumen de los parámetros entrenados.
    """
    if not os.path.exists(model_path):
        error_msg = f"El archivo '{model_path}' no existe."
        logger.error(error_msg)
        return error_msg

    try:
        checkpoint = torch.load(model_path, map_location='cpu')

        info = (
            f"Modelo: {checkpoint.get('model_type', 'Desconocido')}\n"
            f"\tTarget: {checkpoint.get('target_name', 'Desconocido')}\n"
            f"\tÉpocas entrenadas: {checkpoint.get('epochs_trained', 'Desconocido')}\n"
            f"\n"
            f"Dimensiones:\n"
            f"\tInput dim: {checkpoint.get('input_dim', 'Desconocido')}\n"
            f"\tEdge dim: {checkpoint.get('edge_dim', 'Desconocido')}\n"
            f"\tAtom emb %: {checkpoint.get('atom_emb_dim', 'Desconocido')}\n"
            f"\tHybrid emb %: {checkpoint.get('hibrid_emb_dim', 'Desconocido')}\n"
            f"\tBond emb %: {checkpoint.get('bond_emb_dim', 'Desconocido')}\n"
            f"\n"
            f"Hiperparámetros:\n"
            f"\tHidden dim: {checkpoint.get('hidden_dim', 'Desconocido')}\n"
            f"\tNúmero de capas: {checkpoint.get('num_layers', 'Desconocido')}\n"
            f"\tBatch size: {checkpoint.get('batch_size', 'Desconocido')}\n"
            f"\tLearning rate: {checkpoint.get('learning_rate', 'Desconocido')}\n"
            f"\tValid split: {checkpoint.get('valid_split', 'Desconocido')}\n"
            f"\tEarly stopping paciencia: {checkpoint.get('early_stopping_patience', 'No especificada')}\n"
            f"\n"
            f"Transfer mode: {checkpoint.get('transfer_mode', 'No especificado')}"
        )

        return info

    except Exception as e:
        error_msg = f"Error al consultar parámetros del modelo: {str(e)}"
        logger.error(error_msg)
        return error_msg


logger = logging.getLogger(__name__)

def test_all_models_in_directory(models_dir, sdf_dir, targets_file):
    """
    Testea todos los modelos de un directorio con un conjunto de moléculas y targets.
    Genera resultados individuales por modelo y además un archivo resumen CSV con los RMSE
    y el Pearson coefficient, ordenado alfabéticamente por nombre de modelo.
    """
    resumen_file_name = f"resumen_metrics_{os.path.basename(models_dir)}.csv"
    resumen_path = os.path.join(RESULTADOS_DIR, resumen_file_name)
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    resultados = []

    for fname in os.listdir(models_dir):
        model_path = os.path.join(models_dir, fname)

        if not os.path.isfile(model_path):
            continue
        if not fname.endswith((".pt", ".pth")):
            continue  # saltar archivos que no sean modelos

        try:
            # Cargar modelo
            model, device, target_name = cargar_modelo(model_path)

            # Leer datos
            target_dict = read_targets(targets_file)
            data_list = load_data_from_sdf(sdf_dir, target_dict)

            y_true, y_pred = [], []
            for data in data_list:
                data = data.to(device)
                batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model(data.x, data.edge_index, data.edge_attr, batch)
                    pred = out.squeeze().item()
                y_pred.append(pred)
                y_true.append(data.y.item())

            # Calcular RMSE
            rmse = sqrt(mean_squared_error(y_true, y_pred))

            # Calcular Pearson coefficient
            if len(y_true) > 1:  # necesario para scipy
                pearson_r, _ = pearsonr(y_true, y_pred)
            else:
                pearson_r = float("nan")

            # Guardar resultados completos (plots y predicciones)
            test_model_on_directory(model_path, sdf_dir, targets_file)

            resultados.append((fname, f"{rmse:.4f}", f"{pearson_r:.4f}"))

        except Exception as e:
            logger.exception(f"Error con el modelo {fname}")
            resultados.append((fname, f"ERROR ({str(e)})", "ERROR"))

    # Ordenar alfabéticamente por nombre de modelo
    resultados.sort(key=lambda x: x[0].lower())

    # Guardar CSV con RMSE y Pearson
    with open(resumen_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Modelo", "RMSE", "Pearson"])
        for row in resultados:
            writer.writerow(row)

    logger.info(f"Resumen CSV guardado en: {resumen_path}")

