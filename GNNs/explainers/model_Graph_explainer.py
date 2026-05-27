# model_explainer.py
from GNNs.model_tester import cargar_modelo, predecir_molecula
import torch
import torch.nn as nn
import numpy as np
import sys
import logging
from GNNs.data_processing import onehot_to_indices, indices_to_onehot
from GNNs.explainers.explanation_helper import ( 
    tensor_to_abs_numpy, sort_features, normalizar_por_l2, normalizar_por_l1, normalizar_por_maximo, procesar_features_onehot, procesar_features_ordenadas)
from GNNs.explainers.graph_explainer_onehot import generate_perturbed_samples, graph_feature_distance_list
from ui.utils.constants import (
    EDGE_FEATURE_NAMES_EMBEDDING, NODE_FEATURES_NAMES_EMBEDDING)
from GNNs.explainers.graph_explainer_onehot import obtener_argmin

ALGO_NAME = "GraphExplainer"
MININICIAL = sys.float_info.max

logger = logging.getLogger(__name__)

def obtener_graph_explainer(
        checkpoint_path,
        data_indices, 
        target_data_path=None, 
        feature_mask=[1, 1, 1, 1, 1, 1], 
        num_samples=50, 
        noise_level=0.05, 
        device='cpu',
        batch_mode = False):
    
    muestra = indices_to_onehot(data_indices)

    # --- 1. OBTENER INFORMACIÓN REAL ---
    real_val = data_indices.y
    mol_name = data_indices.name
    
    # Generar muestras perturbadas
    perturbed_samples = generate_perturbed_samples(muestra, feature_mask, num_samples, noise_level)
    perturbed_samples_embedding = []

    # Obtener modelo
    model, device, target_name = cargar_modelo(checkpoint_path)

    muestra_for_model = onehot_to_indices(muestra.to(device))
    prediccion_original = predecir_molecula(model, muestra_for_model, device)

    # Predecir las muestras perturbadas
    predicciones_perturbadas = []
    for perturbed in perturbed_samples:
        perturbed = perturbed.to(device)
        perturbed_for_model = onehot_to_indices(perturbed)  # <-- aquí el puente
        pred = predecir_molecula(model, perturbed_for_model, device)
        predicciones_perturbadas.append(pred)
        perturbed_samples_embedding.append(perturbed_for_model)

    # Convertir a tensor [num_samples,1]
    predicciones_perturbadas = torch.tensor(predicciones_perturbadas, dtype=torch.float, device=device).unsqueeze(1)

    # Calcular distancias entre la muestra original y las perturbaciones
    feature_distances = graph_feature_distance_list(muestra_for_model, perturbed_samples_embedding)

    # Obtener E
    E_list = [data_z.x.to(device) for data_z in perturbed_samples_embedding]

    # Lo mismo con los edges
    A_list = []
    for data_z in perturbed_samples_embedding:
    # for data_z in perturbed_samples:
        if data_z.edge_attr is not None:
            A_list.append(data_z.edge_attr.to(device))

    # --------------- HACERLO CON OPTIMIZACION DE PYTORCH ----------------------
    alfa, beta, gamma, delta, loss = obtener_argmin(feature_distances, predicciones_perturbadas, E_list, A_list)
    # Verificar que aprendimos algo distinto de cero
    print(f"Max Alfa: {alfa.max().item():.4f}, Min Alfa: {alfa.min().item():.4f}")
    print(f"Max Beta: {beta.max().item():.4f}, Min Beta: {beta.min().item():.4f}")

    # ==========================================================================
    # PROCESAMIENTO DE MATRICES
    # ==========================================================================

    # ====================================================================
    # NORMALIZACIÓN GLOBAL
    # ====================================================================
    
    # 1. Convertir todos los tensores a NumPy (Valores absolutos crudos)
    alfa_raw = tensor_to_abs_numpy(alfa)
    beta_raw = tensor_to_abs_numpy(beta)
    gamma_raw = tensor_to_abs_numpy(gamma) if gamma is not None else np.array([])
    delta_raw = tensor_to_abs_numpy(delta) if delta is not None else np.array([])

    # 2. Encontrar el MÁXIMO GLOBAL entre todos los arreglos
    arreglos_validos = [arr for arr in (alfa_raw, beta_raw, gamma_raw, delta_raw) if arr is not None and arr.size > 0]
    
    if arreglos_validos:
        max_global = max([np.max(arr) for arr in arreglos_validos])
    else:
        max_global = 1.0
        
    if max_global == 0:
        max_global = 1.0 # Seguridad para evitar división por cero
        
    # 3. Aplicar la normalización global a todos (Manteniendo la escala relativa)
    alfa_norm = alfa_raw / max_global if alfa_raw is not None else None
    beta_norm = beta_raw / max_global if beta_raw is not None else None
    gamma_norm = gamma_raw / max_global if gamma_raw.size > 0 else np.array([])
    delta_norm = delta_raw / max_global if delta_raw.size > 0 else np.array([])

    # ====================================================================
    # PROCESAMIENTO Y ORDENAMIENTO (Usando los arrays ya normalizados)
    # ====================================================================

    # 1. ALFA (Node Features)
    node_feature_names = NODE_FEATURES_NAMES_EMBEDDING
    alfa_sorted, row_labels_alfa = sort_features(alfa_norm, node_feature_names)

    # 2. GAMMA (Edge Features)
    if muestra.edge_attr is not None and gamma_norm.size > 0:
        edge_feature_names = EDGE_FEATURE_NAMES_EMBEDDING
        gamma_sorted, row_labels_gamma = sort_features(gamma_norm, edge_feature_names)
    else:
        gamma_sorted = np.array([])
        row_labels_gamma = []

    # Preparamos el nombre del modelo para la carpeta
    model_folder_name = checkpoint_path.split('/')[-1].split('.')[0]

    # Retorno en Batch Mode
    if batch_mode:
        return {
            'mol_name': mol_name, 
            'alfa': alfa_sorted,  
            'beta': beta_norm,    # Los nodos no se ordenan aquí, los ordenas en tu otro script
            'gamma': gamma_sorted,
            'delta': delta_norm   # Los enlaces tampoco
        }
        
    return 0

# --- HELPER: STACK Y NORMALIZACIÓN GLOBAL ---
def stack_and_normalize(tensor_list, device):
    if not tensor_list: return None, 1.0
    
    # 1. Stack: [Samples, N_elementos, Features]
    #    Asumimos que N es constante (LIME estándar). 
    stacked = torch.stack(tensor_list).to(device)
    
    # 2. Calcular Min/Max Global
    #    Aplanamos (Sample y N) para buscar el min/max de cada feature en todo el dataset
    flattened = stacked.view(-1, stacked.shape[-1])
    
    val_min = flattened.min(dim=0).values
    val_max = flattened.max(dim=0).values
    val_range = val_max - val_min
    val_range[val_range == 0] = 1.0 # Evitar división por 0
    
    # 3. Normalizar
    #    [S, N, F] - [F] funciona directo
    normalized_stacked = (stacked - val_min) / val_range
    
    # Factor de escala para la predicción (1 / total_elementos)
    # N_elementos = stacked.shape[1]
    # scale = 1.0 / N_elementos
    
    return normalized_stacked