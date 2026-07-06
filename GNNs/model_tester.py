#model_tester.py
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from GNNs.model_trainer import create_model, calc_dim
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from GNNs.data_processing import read_targets, load_data_from_sdf, mol_to_graph_data
import logging
import sys
from ui.utils.constants import RESULTADOS_DIR, hybridization_types, periodic_elements, N_BOND_TYPES, OTHER_EDGE_FEATURES, OTHER_NODE_FEATURES
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

        # Nombre base de archivos (carpeta de origen)
        folder_name = os.path.basename(sdf_dir.rstrip(os.sep))

        # --- CAMBIO AQUÍ: Definir ruta CSV y llamar a la función auxiliar ---
        output_csv_path = os.path.join(
            model_results_dir,
            f"predicciones_{model_name_no_ext}_{folder_name}.csv"
        )
        
        # Llamamos a la función auxiliar que definimos arriba
        guardar_predicciones_csv(output_csv_path, filenames, y_true, y_pred)
        
        logger.info(f"Predicciones guardadas en CSV: {output_csv_path}")
        # -------------------------------------------------------------------

        # RMSE
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        logger.info(f"RMSE: {rmse:.4f}")

        # R2 score
        r2 = r2_score(y_true, y_pred)
        logger.info(f"R2 score: {r2:.4f}")

        # Pearson coefficient
        if len(y_true) > 1:
            pearson_r, _ = pearsonr(y_true, y_pred)
            logger.info(f"Pearson coefficient: {pearson_r:.4f}")
        else:
            pearson_r = float("nan")
            logger.info("Pearson coefficient: No se puede calcular con un solo punto.")

        # Scatter plot
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.7)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
        plt.xlabel("Real Solubility", fontsize = 20)
        plt.ylabel("Predicted Solubility", fontsize = 20)
        plt.tick_params(axis='both', which='major', labelsize=16)
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

        return plot_filename

    except Exception as e:
        raise ValueError(e)
    
def guardar_predicciones_csv(ruta_salida, nombres, y_real, y_pred):
    """
    Guarda los resultados de la inferencia en un archivo CSV incluyendo el error absoluto.
    """
    # Crear un DataFrame con los datos básicos
    df = pd.DataFrame({
        'Molecula': nombres,
        'Solubilidad_Real': y_real,
        'Solubilidad_Predicha': y_pred
    })
    
    # Calcular el error absoluto: |Real - Predicho|
    df['Error_Absoluto'] = (df['Solubilidad_Real'] - df['Solubilidad_Predicha']).abs()
    
    # Guardar a CSV sin incluir el índice numérico de pandas
    df.to_csv(ruta_salida, index=False, float_format='%.4f')


def obtener_info_checkpoint(model_path):
    """
    Carga un checkpoint y devuelve un resumen de los parámetros entrenados.
    """
    if not os.path.exists(model_path):
        error_msg = f"El archivo '{model_path}' no existe."
        logger.exception(error_msg)
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
        logger.exception(error_msg)
        return error_msg


logger = logging.getLogger(__name__)

def test_all_models_in_directory(models_dir, sdf_dir, targets_file):
    """
    Testea todos los modelos de un directorio con un conjunto de moléculas y targets.
    Genera resultados individuales por modelo y además un archivo resumen CSV con los RMSE,
    Pearson coefficient y R2, ordenado alfabéticamente por nombre de modelo.
    """
    resumen_file_name = f"resumen_metrics_{os.path.basename(models_dir)}.csv"
    resumen_path = os.path.join(RESULTADOS_DIR, resumen_file_name)
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    # Leer datos
    target_dict = read_targets(targets_file)
    data_list = load_data_from_sdf(sdf_dir, target_dict)

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

            y_true, y_pred = [], []
            for data in data_list:
                data = data.to(device)
                
                # Crear batch ficticio si es una sola molécula
                batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
                
                with torch.no_grad():
                    out = model(data.x, data.edge_index, data.edge_attr, batch)
                    pred = out.squeeze().item()
                
                y_pred.append(pred)
                y_true.append(data.y.item())

            # --------------------------
            # CÁLCULO DE MÉTRICAS
            # --------------------------
            
            # 1. RMSE
            rmse = sqrt(mean_squared_error(y_true, y_pred))

            # 2. Pearson y R2
            if len(y_true) > 1:
                pearson_r, _ = pearsonr(y_true, y_pred)
                r2_val = r2_score(y_true, y_pred)
            else:
                pearson_r = float("nan")
                r2_val = float("nan")

            # Guardar resultados completos (plots y predicciones)
            # Asumo que esta función ya hace sus propios guardados
            test_model_on_directory(model_path, sdf_dir, targets_file)

            # Agregamos R2 a la tupla de resultados
            resultados.append((fname, f"{rmse:.4f}", f"{pearson_r:.4f}", f"{r2_val:.4f}"))

        except Exception as e:
            logger.exception(f"Error con el modelo {fname}")
            # Mantenemos la consistencia de columnas en caso de error (4 columnas)
            resultados.append((fname, f"ERROR ({str(e)})", "ERROR", "ERROR"))

    # Ordenar alfabéticamente por nombre de modelo
    resultados.sort(key=lambda x: x[0].lower())

    # Guardar CSV con RMSE, Pearson y R2
    with open(resumen_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Agregamos la cabecera R2
        writer.writerow(["Modelo", "RMSE", "Pearson", "R2"]) 
        for row in resultados:
            writer.writerow(row)

    logger.info(f"Resumen CSV guardado en: {resumen_path}")

    return resumen_path  # <--- AGREGA ESTO AL FINAL

def test_all_splits(
    models_mother_dir, 
    data_mother_dir, 
    targets_file, 
    base_results_dir = RESULTADOS_DIR,
    test_folder_name="test" # Puede ser "validation" o "test", según cómo se llame tu carpeta de SDFs
):
    """
    Explora la carpeta madre de modelos, busca los splits y los empareja con
    los datos correspondientes en la carpeta madre de datos para testearlos.
    """
    # 1. Obtener las subcarpetas de los splits (ej: split_0, split_1...)
    splits_modelos = [d for d in os.listdir(models_mother_dir) if os.path.isdir(os.path.join(models_mother_dir, d))]
    
    for split_folder in sorted(splits_modelos):
        models_dir = os.path.join(models_mother_dir, split_folder)
        
        # 2. Buscar la carpeta de datos correspondiente
        # Ej: datos_madre/split_0/test
        sdf_dir = os.path.join(data_mother_dir, split_folder, test_folder_name)
        
        if not os.path.exists(sdf_dir):
            logging.warning(f"Ignorando '{split_folder}': No se encontró la carpeta de datos en {sdf_dir}")
            continue
            
        logging.info(f"\n{'='*50}\nIniciando Testing para: {split_folder}\n{'='*50}")
        
        # 3. Crear directorio de resultados para este split
        split_results_dir = os.path.join(base_results_dir, split_folder)
        os.makedirs(split_results_dir, exist_ok=True)
        
        # 4. Lanzar tu función de testeo
        test_all_models_in_directory(
            models_dir=models_dir,
            sdf_dir=sdf_dir,
            targets_file=targets_file,
            output_dir=base_results_dir # <--- Pasamos la ruta de guardado
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    # Rutas principales
    MODELOS_MADRE = "/home/andromeda/Documentos/Philippe/TrabajoInvestigacion/Modelos/PruebaOneHot"      # Donde están las carpetas split_0, split_1... con los .pt del modelo
    RESULTADOS_MADRE = "/home/andromeda/Documentos/Philippe/TrabajoInvestigacion/Modelos/PruebOneHot"   # Donde se guardarán los CSV finales
    
    print("🚀 Iniciando el testing masivo de todos los splits...")
    
    # -------------------TESTEO CON SDF ------------------------------
    DATOS_MADRE = "/home/andromeda/Documentos/Philippe/Datos Philippe/SplitsSMILES" # Donde están las carpetas de los splits con los SDF
    TARGETS = "/home/andromeda/Documentos/Philippe/Datos Philippe/Splits/pIC50.txt"
    test_all_splits(
        models_mother_dir=MODELOS_MADRE,
        data_mother_dir=DATOS_MADRE,
        targets_file=TARGETS,
        base_results_dir=RESULTADOS_MADRE,
        test_folder_name="test" # Cambia esto a "test" si tu carpeta de datos a probar se llama así
    )

    # -------------------TESTEO CON .PT ------------------------------
    # DATOS_MADRE = "/home/andromeda/Documentos/Philippe/Datos Philippe/SplitsSMILES" # CARPETA DONDE ESTAN LOS .PT
    # test_all_splits_pt(
    #     models_mother_dir=MODELOS_MADRE,
    #     data_mother_dir=DATOS_MADRE,
    #     base_results_dir=RESULTADOS_MADRE
    # )
    
    print("✅ ¡Testing finalizado!")