#model_tester.py
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from collections import defaultdict
import logging
import sys
import csv
from scipy.stats import pearsonr

dir_actual = os.path.dirname(os.path.abspath(__file__))
dir_padre = os.path.abspath(os.path.join(dir_actual, ".."))
sys.path.insert(0, dir_padre)

from GNNs.data_processing import read_targets, load_data_from_sdf, mol_to_graph_data
from GNNs.model_trainer import create_model, calc_dim
from ui.utils.constants import *


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
    calc_non_cov_emb_dim = calc_dim(N_NON_COV * NON_COV_EMB_PR)

    embeddings = [calc_atom_emb_dim, calc_hibrid_emb_dim, calc_bond_emb_dim, calc_non_cov_emb_dim] 

    # 2. Dimensión Final de los NODOS (x)
    # Suma: Emb(Átomo) + Emb(Hibridación) + Continuas(Degree, Total_Hs, Aromatic, Donor, Acceptor)
    input_dim = calc_atom_emb_dim + calc_hibrid_emb_dim + OTHER_NODE_FEATURES

    # 3. Dimensión Final de los ENLACES (edge_attr)
    # Suma: Emb(Covalente) + Emb(No Covalente) + Continuas(Distancia, Flexibilidad)
    # Como la capa de Embedding ya comprimió las 25 binarias, las únicas continuas puras que sobran son 2
    edge_dim = calc_bond_emb_dim + calc_non_cov_emb_dim + OTHER_EDGE_FEATURES

    # Crear modelo con los parámetros guardados
    model = create_model(
        model_type,
        input_dim,
        embeddings, 
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
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=True)
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

def test_model_on_directory(checkpoint_path, sdf_dir, targets_file, output_dir=RESULTADOS_DIR):
    """
    Testea un ÚNICO modelo en un directorio de datos específico.
    Calcula métricas, guarda predicciones en CSV y genera el scatter plot.
    """
    try:
        model, device, target_name = cargar_modelo(checkpoint_path)

        # Crear carpetas para los resultados de este modelo
        model_filename = os.path.basename(checkpoint_path)
        model_name_no_ext = os.path.splitext(model_filename)[0]
        
        model_results_dir = os.path.join(output_dir, model_name_no_ext)
        os.makedirs(model_results_dir, exist_ok=True)

        # Leer datos
        target_dict = read_targets(targets_file)
        data_list = load_data_from_sdf(sdf_dir, target_dict)

        y_true, y_pred, filenames = [], [], []

        # Inferencia
        for data in data_list:
            data = data.to(device)
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

            with torch.no_grad():
                out = model(data.x, data.edge_index, data.edge_attr, batch)
                pred = out.squeeze().item()

            y_pred.append(pred)
            y_true.append(data.y.item())
            filenames.append(data.name if hasattr(data, 'name') else 'unknown')

        folder_name = os.path.basename(sdf_dir.rstrip(os.sep))

        # Guardar predicciones en CSV
        output_csv_path = os.path.join(
            model_results_dir,
            f"predicciones_{model_name_no_ext}_{folder_name}.csv"
        )
        guardar_predicciones_csv(output_csv_path, filenames, y_true, y_pred)
        logging.info(f"Predicciones guardadas en CSV: {output_csv_path}")

        # Calcular e imprimir métricas
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        logging.info(f"RMSE: {rmse:.4f}")

        if len(y_true) > 1:
            r2 = r2_score(y_true, y_pred)
            pearson_r, _ = pearsonr(y_true, y_pred)
            logging.info(f"R2 score: {r2:.4f} | Pearson: {pearson_r:.4f}")
        else:
            logging.info("R2/Pearson: No se pueden calcular con un solo punto.")

        # --- Generar el scatter plot ---
        # Llamamos a la función genérica que creamos antes
        plot_filename = generar_scatter_plot(
            y_true=y_true, 
            y_pred=y_pred, 
            model_results_dir=model_results_dir, 
            model_name_no_ext=model_name_no_ext, 
            folder_name=folder_name
        )

        return plot_filename

    except Exception as e:
        logging.error(f"Error testeando el modelo {checkpoint_path}: {e}")
        raise ValueError(e)
    

def test_model_on_directory_pt(checkpoint_path, pt_file, output_dir=RESULTADOS_DIR):
    """
    Testea un ÚNICO modelo en un archivo de datos .pt específico.
    Calcula métricas, guarda predicciones en CSV y genera el scatter plot.
    """
    try:
        model, device, target_name = cargar_modelo(checkpoint_path)

        # Crear carpetas para los resultados de este modelo
        model_filename = os.path.basename(checkpoint_path)
        model_name_no_ext = os.path.splitext(model_filename)[0]
        
        model_results_dir = os.path.join(output_dir, model_name_no_ext)
        os.makedirs(model_results_dir, exist_ok=True)

        # Leer datos
        data_list = torch.load(pt_file)

        if not data_list or not isinstance(data_list, list):
            raise ValueError("El archivo .pt está vacío o no contiene una lista válida.")

        y_true, y_pred, filenames = [], [], []

        # Inferencia
        for data in data_list:
            data = data.to(device)
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

            with torch.no_grad():
                out = model(data.x, data.edge_index, data.edge_attr, batch)
                pred = out.squeeze().item()

            y_pred.append(pred)
            y_true.append(data.y.item())
            filenames.append(data.name if hasattr(data, 'name') else 'unknown')

        # Nombre base para identificar el set de datos (sin extensión .pt)
        folder_name = os.path.splitext(os.path.basename(pt_file))[0]

        # Guardar predicciones en CSV
        output_csv_path = os.path.join(
            model_results_dir,
            f"predicciones_{model_name_no_ext}_{folder_name}.csv"
        )
        guardar_predicciones_csv(output_csv_path, filenames, y_true, y_pred)
        logging.info(f"Predicciones guardadas en CSV: {output_csv_path}")

        # Calcular e imprimir métricas
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        logging.info(f"RMSE: {rmse:.4f}")

        if len(y_true) > 1:
            r2 = r2_score(y_true, y_pred)
            pearson_r, _ = pearsonr(y_true, y_pred)
            logging.info(f"R2 score: {r2:.4f} | Pearson: {pearson_r:.4f}")
        else:
            logging.info("R2/Pearson: No se pueden calcular con un solo punto.")

        # --- Generar el scatter plot ---
        # Llamamos a la función genérica pasando los argumentos de forma explícita
        plot_filename = generar_scatter_plot(
            y_true=y_true, 
            y_pred=y_pred, 
            model_results_dir=model_results_dir, 
            model_name_no_ext=model_name_no_ext, 
            folder_name=folder_name
        )

        return plot_filename

    except Exception as e:
        logging.error(f"Error testeando el modelo {checkpoint_path}: {e}")
        raise ValueError(e)
    
def generar_scatter_plot(y_true, y_pred, model_results_dir, model_name_no_ext, folder_name, tolerance=1.0):
    """
    Genera el scatter plot con zona de tolerancia, sin leyenda y con la diagonal punteada.
    """
    plt.figure(figsize=(6, 6))
    
    # Dibujar los puntos
    plt.scatter(y_true, y_pred, alpha=0.7)
    
    # Rango para la diagonal
    x_min, x_max = min(y_true), max(y_true)
    x_range = np.array([x_min, x_max])
    
    # Línea diagonal central (AHORA PUNTEADA Y SIN LABEL)
    plt.plot(x_range, x_range, color='red', linestyle='--', linewidth=2)
    
    # Zona de tolerancia sombreada
    plt.fill_between(
        x_range, 
        x_range - tolerance, 
        x_range + tolerance, 
        color='gray', 
        alpha=0.2
    )
    
    # Líneas de umbral (bordes del sombreado)
    plt.plot(x_range, x_range + tolerance, color='gray', linestyle=':', linewidth=1, alpha=0.4)
    plt.plot(x_range, x_range - tolerance, color='gray', linestyle=':', linewidth=1, alpha=0.4)

    # Estética
    plt.xlabel("Real", fontsize=20)
    plt.ylabel("Predicted", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Eliminamos plt.legend() para que no aparezca la guía
    
    plt.tight_layout()

    # Ruta de guardado
    plot_filename = os.path.join(
        model_results_dir,
        f"scatter_plot_{model_name_no_ext}_{folder_name}.png"
    )
    plt.savefig(plot_filename)
    plt.close()

    logger.info(f"Scatter plot guardado en: {plot_filename}")
    return plot_filename
    
def guardar_predicciones_csv(ruta_salida, nombres, y_real, y_pred):
    """
    Guarda los resultados de la inferencia en un archivo CSV incluyendo el error absoluto.
    """
    # Crear un DataFrame con los datos básicos
    df = pd.DataFrame({
        'Molecula': nombres,
        'Real': y_real,
        'Predicha': y_pred
    })
    
    # Calcular el error absoluto: |Real - Predicho|
    df['Error_Absoluto'] = (df['Real'] - df['Predicha']).abs()
    
    # Guardar a CSV sin incluir el índice numérico de pandas
    df.to_csv(ruta_salida, index=False, float_format='%.4f')


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

def test_all_models_in_directory(models_dir, sdf_dir, targets_file, output_dir = None, acumulador=None):
    """
    Testea los modelos de un directorio. Guarda un CSV de resumen de métricas, 
    un CSV de predicciones por modelo, y alimenta el acumulador global (opcional).
    """
    basename = f"{os.path.basename(models_dir)}"
    resumen_file_name = f"resumen_metrics_{basename}.csv"

    # Esto es para que si es un split me ponga los csv fuera de la carpeta, sino dentro :p
    if output_dir is None:
        os.makedirs(RESULTADOS_DIR, exist_ok=True)
        output_dir = os.path.join(RESULTADOS_DIR, basename)
        os.makedirs(output_dir, exist_ok=True)
        resumen_path = os.path.join(output_dir, resumen_file_name) 
    else:
        parent_dir = os.path.dirname(os.path.abspath(output_dir))
        resumen_path = os.path.join(parent_dir, resumen_file_name)
    
    target_dict = read_targets(targets_file)
    data_list = load_data_from_sdf(sdf_dir, target_dict)

    resultados_resumen = []

    for fname in os.listdir(models_dir):
        model_path = os.path.join(models_dir, fname)

        if not os.path.isfile(model_path) or not fname.endswith((".pt", ".pth")):
            continue  

        try:
            model, device, target_name = cargar_modelo(model_path)

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

            # 1. Calcular Métricas
            rmse = sqrt(mean_squared_error(y_true, y_pred))
            if len(y_true) > 1:
                pearson_r, _ = pearsonr(y_true, y_pred)
                r2_val = r2_score(y_true, y_pred)
            else:
                pearson_r, r2_val = float("nan"), float("nan")

            resultados_resumen.append((fname, f"{rmse:.4f}", f"{pearson_r:.4f}", f"{r2_val:.4f}"))

            # 2. Guardar CSV de predicciones
            model_name_no_ext = os.path.splitext(fname)[0]
            output_csv_path = os.path.join(output_dir, f"predicciones_{model_name_no_ext}_{basename}.csv")
            guardar_predicciones_csv(output_csv_path, filenames, y_true, y_pred)

            # 3. ALIMENTAR EL ACUMULADOR GLOBAL (Solo si se proporcionó uno)
            if acumulador is not None:
                acumulador[fname]["y_true"].extend(y_true)
                acumulador[fname]["y_pred"].extend(y_pred)

        except Exception as e:
            logging.exception(f"Error con el modelo {fname}: {e}")
            resultados_resumen.append((fname, f"ERROR ({str(e)})", "ERROR", "ERROR"))

    # Guardar CSV Resumen
    resultados_resumen.sort(key=lambda x: x[0].lower())
    with open(resumen_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Modelo", "RMSE", "Pearson", "R2"]) 
        for row in resultados_resumen:
            writer.writerow(row)

    logging.info(f"Resumen CSV guardado en: {resumen_path}")

def test_all_models_in_directory_pt(models_dir, pt_file, output_dir=None, acumulador=None):
    """
    Testea los modelos de un directorio usando datos de un archivo .pt.
    Guarda un CSV de resumen de métricas, un CSV de predicciones por modelo, 
    y alimenta el acumulador global (opcional).
    """
    basename = f"{os.path.basename(models_dir)}"
    resumen_file_name = f"resumen_metrics_{basename}.csv"

    # Esto es para que si es un split me ponga los csv fuera de la carpeta, sino dentro :p
    if output_dir is None:
        os.makedirs(RESULTADOS_DIR, exist_ok=True)
        output_dir = os.path.join(RESULTADOS_DIR, basename)
        os.makedirs(output_dir, exist_ok=True)
        resumen_path = os.path.join(output_dir, resumen_file_name) 
    else:
        parent_dir = os.path.dirname(os.path.abspath(output_dir))
        resumen_path = os.path.join(parent_dir, resumen_file_name)
    
    # 1. Cargar la lista de moléculas directamente desde .pt
    data_list = torch.load(pt_file)

    if not data_list or not isinstance(data_list, list):
        raise ValueError("El archivo .pt está vacío o no contiene una lista válida.")

    resultados_resumen = []

    for fname in os.listdir(models_dir):
        model_path = os.path.join(models_dir, fname)

        if not os.path.isfile(model_path) or not fname.endswith((".pt", ".pth")):
            continue  

        try:
            # Cargar modelo
            model, device, target_name = cargar_modelo(model_path)

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

            # 1. Calcular Métricas
            rmse = sqrt(mean_squared_error(y_true, y_pred))
            if len(y_true) > 1:
                pearson_r, _ = pearsonr(y_true, y_pred)
                r2_val = r2_score(y_true, y_pred)
            else:
                pearson_r, r2_val = float("nan"), float("nan")

            resultados_resumen.append((fname, f"{rmse:.4f}", f"{pearson_r:.4f}", f"{r2_val:.4f}"))

            # 2. Guardar CSV de predicciones
            model_name_no_ext = os.path.splitext(fname)[0]
            output_csv_path = os.path.join(output_dir, f"predicciones_{model_name_no_ext}_{basename}.csv")
            guardar_predicciones_csv(output_csv_path, filenames, y_true, y_pred)

            # 3. ALIMENTAR EL ACUMULADOR GLOBAL (Solo si se proporcionó uno)
            if acumulador is not None:
                acumulador[fname]["y_true"].extend(y_true)
                acumulador[fname]["y_pred"].extend(y_pred)

        except Exception as e:
            logging.exception(f"Error con el modelo {fname}: {e}")
            resultados_resumen.append((fname, f"ERROR ({str(e)})", "ERROR", "ERROR"))

    # Guardar CSV Resumen
    resultados_resumen.sort(key=lambda x: x[0].lower())
    with open(resumen_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Modelo", "RMSE", "Pearson", "R2"]) 
        for row in resultados_resumen:
            writer.writerow(row)

    logging.info(f"Resumen CSV guardado en: {resumen_path}")

def test_all_splits(
    models_mother_dir, 
    data_mother_dir, 
    targets_file, 
    base_results_dir=RESULTADOS_DIR,
    test_folder_name="test"
):
    """
    Explora la carpeta madre de modelos, busca los splits, testea los modelos
    y genera un único gráfico consolidado por modelo con todos los splits.
    """
    splits_modelos = [d for d in os.listdir(models_mother_dir) if os.path.isdir(os.path.join(models_mother_dir, d))]
    
    # --- NUEVO: ACUMULADOR GLOBAL ---
    # Diccionario con formato: { "nombre_modelo.pt": {"y_true": [], "y_pred": []} }
    predicciones_globales = defaultdict(lambda: {"y_true": [], "y_pred": []})
    
    for split_folder in sorted(splits_modelos):
        models_dir = os.path.join(models_mother_dir, split_folder)
        sdf_dir = os.path.join(data_mother_dir, split_folder, test_folder_name)
        
        if not os.path.exists(sdf_dir):
            logging.warning(f"Ignorando '{split_folder}': No se encontró la carpeta de datos en {sdf_dir}")
            continue
            
        logging.info(f"\n{'='*50}\nIniciando Testing para: {split_folder}\n{'='*50}")
        
        split_results_dir = os.path.join(base_results_dir, split_folder)
        os.makedirs(split_results_dir, exist_ok=True)
        
        # Ejecutamos el testeo y pasamos el acumulador global
        test_all_models_in_directory(
            models_dir=models_dir,
            sdf_dir=sdf_dir,
            targets_file=targets_file,
            output_dir=split_results_dir,
            acumulador=predicciones_globales # <--- Pasamos el diccionario
        )

    # --- NUEVO: GENERAR GRÁFICOS CONSOLIDADOS AL FINAL ---
    logging.info(f"\n{'='*50}\nGenerando Gráficos Consolidados\n{'='*50}")
    
    # Creamos una carpeta para los plots globales
    plots_dir = os.path.join(base_results_dir, "plots_globales")
    os.makedirs(plots_dir, exist_ok=True)

    for model_filename, datos in predicciones_globales.items():
        if not datos["y_true"]: # Si por alguna razón está vacío, saltar
            continue
            
        model_name_no_ext = os.path.splitext(model_filename)[0]
        
        # Llamamos a tu función de ploteo pasando TODOS los datos acumulados
        generar_scatter_plot(
            y_true=datos["y_true"],
            y_pred=datos["y_pred"],
            model_results_dir=plots_dir,
            model_name_no_ext=model_name_no_ext,
            folder_name="todos_los_splits"
        )

def test_all_splits_pt(
    models_mother_dir, 
    data_mother_dir, 
    base_results_dir=RESULTADOS_DIR,
    test_file_prefix="pocket_BD_test_" 
):
    """
    Explora la carpeta madre de modelos, identifica los splits, busca el archivo 
    .pt de testeo correspondiente, lanza la evaluación y genera un gráfico consolidado.
    """
    # 1. Obtener las subcarpetas de los splits de modelos (ej: split_0, split_1...)
    splits_modelos = [d for d in os.listdir(models_mother_dir) if os.path.isdir(os.path.join(models_mother_dir, d))]
    
    # --- NUEVO: ACUMULADOR GLOBAL ---
    # Diccionario con formato: { "nombre_modelo.pt": {"y_true": [], "y_pred": []} }
    predicciones_globales = defaultdict(lambda: {"y_true": [], "y_pred": []})
    
    for split_folder in sorted(splits_modelos):
        models_dir = os.path.join(models_mother_dir, split_folder)
        
        # 2. Extraer el número del split del nombre de la carpeta (asumiendo formato "split_X")
        match = re.search(r'_(\d+)$', split_folder)
        if not match:
            logging.warning(f"Ignorando '{split_folder}': No se pudo extraer un número de split del nombre.")
            continue
            
        num_split = match.group(1)
        
        # Buscar pocket_BD_test_X.pt o ligand_BD_test_X.pt
        test_candidates = [
            f"pocket_BD_test_{num_split}.pt",
            f"ligand_BD_test_{num_split}.pt"
        ]

        pt_file_path = None

        for filename in test_candidates:
            candidate = os.path.join(data_mother_dir, filename)
            if os.path.exists(candidate):
                pt_file_path = candidate
                test_pt_filename = filename
                break

        if pt_file_path is None:
            logging.warning(
                f"Ignorando '{split_folder}': No se encontró ningún archivo de test para el split {num_split}"
            )
            continue
            
        logging.info(f"\n{'='*50}\nIniciando Testing para: {split_folder} con {test_pt_filename}\n{'='*50}")
        
        # 4. Crear directorio de resultados para este split
        split_results_dir = os.path.join(base_results_dir, split_folder)
        os.makedirs(split_results_dir, exist_ok=True)
        
        # 5. Lanzar tu función de testeo pasando el acumulador global
        test_all_models_in_directory_pt(
            models_dir=models_dir,
            pt_file=pt_file_path,
            output_dir=split_results_dir,
            acumulador=predicciones_globales # <--- Pasamos el diccionario
        )

    # --- NUEVO: GENERAR GRÁFICOS CONSOLIDADOS AL FINAL ---
    logging.info(f"\n{'='*50}\nGenerando Gráficos Consolidados\n{'='*50}")
    
    # Creamos una carpeta para los plots globales
    plots_dir = os.path.join(base_results_dir, "plots_globales")
    os.makedirs(plots_dir, exist_ok=True)

    for model_filename, datos in predicciones_globales.items():
        if not datos["y_true"]: # Si por alguna razón está vacío, saltar
            continue
            
        model_name_no_ext = os.path.splitext(model_filename)[0]
        
        # Llamamos a tu función de ploteo pasando TODOS los datos acumulados
        generar_scatter_plot(
            y_true=datos["y_true"],
            y_pred=datos["y_pred"],
            model_results_dir=plots_dir,
            model_name_no_ext=model_name_no_ext,
            folder_name="todos_los_splits"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    # Rutas principales
    MODELOS_MADRE = "Modelos/Modelos_25F"      # Donde están las carpetas split_0, split_1... con los .pt del modelo
    RESULTADOS_MADRE = "Resultados/Resultados_25F"   # Donde se guardarán los CSV finales
    
    print("🚀 Iniciando el testing masivo de todos los splits...")
    
    # -------------------TESTEO CON SDF ------------------------------
    # DATOS_MADRE = "/home/philippe/Documents/Databases/URV_database_vNatalia/Splits" # Donde están las carpetas de los splits con los SDF
    # TARGETS = "/home/philippe/Documents/Databases/URV_Database_2025_Octubre/pIC50.txt"
    # test_all_splits(
    #     models_mother_dir=MODELOS_MADRE,
    #     data_mother_dir=DATOS_MADRE,
    #     targets_file=TARGETS,
    #     base_results_dir=RESULTADOS_MADRE,
    #     test_folder_name="test" # Cambia esto a "test" si tu carpeta de datos a probar se llama así
    # )

    # -------------------TESTEO CON .PT ------------------------------
    DATOS_MADRE = "/home/philippe/Documents/Databases/7_A_25Features_PT/pocket_BD/pocket_BD.pt" # CARPETA DONDE ESTAN LOS .PT
    # test_all_splits_pt(
    #     models_mother_dir=MODELOS_MADRE,
    #     data_mother_dir=DATOS_MADRE,
    #     base_results_dir=RESULTADOS_MADRE
    # )

    test_model_on_directory_pt(
        checkpoint_path=MODELOS_MADRE,
        pt_file=DATOS_MADRE,
        output_dir=RESULTADOS_MADRE
    )
    
    print("✅ ¡Testing finalizado!")