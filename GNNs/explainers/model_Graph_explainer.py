# model_explainer.py
from GNNs.model_tester import cargar_modelo, predecir_molecula
import torch
import torch.nn as nn
import numpy as np
import sys
import logging
from GNNs.data_processing import onehot_to_indices, indices_to_onehot
from GNNs.explainers.explanation_helper import ( 
    tensor_to_abs_numpy, normalizar_por_l2, procesar_features_ordenadas)
from GNNs.explainers.graph_explainer_onehot import *
from ui.utils.constants import NODE_FEATURES_NAMES_EMBEDDING, EDGE_FEATURE_NAMES_EMBEDDING

ALGO_NAME = "GraphExplainer"
MININICIAL = sys.float_info.max

logger = logging.getLogger(__name__)

def obtener_graph_explainer_from_pt(
        checkpoint_path,
        data_indices, 
        target_data_path=None, 
        feature_mask=[0, 0, 0, 0, 1, 1, 1, 1], 
        num_samples=500, 
        noise_level=0.2, 
        device='cpu',
        batch_mode = False):
    
    muestra = indices_to_onehot(data_indices)

    # --- 1. OBTENER INFORMACIÓN REAL ---
    real_val = data_indices.y
    mol_name = data_indices.name
    
    # Generar muestras perturbadas
    perturbed_samples = generate_perturbed_samples(muestra, feature_mask=feature_mask, num_samples=num_samples, noise_level=noise_level)
    perturbed_samples_embedding = []

    # Obtener modelo
    model, device, target_name = cargar_modelo(checkpoint_path)

    muestra_for_model = onehot_to_indices(muestra.to(device))
    # prediccion_original = predecir_molecula(model, muestra_for_model, device)

    # 1. Convertir todas las muestras al formato del modelo en una lista
    muestras_procesadas = [onehot_to_indices(p.to(device)) for p in perturbed_samples]
    
    # 2. Unirlas en un solo "Súper Grafo" (Batch)
    batch_data = Batch.from_data_list(muestras_procesadas)
    
    # 3. Hacer UNA SOLA predicción masiva
    with torch.no_grad():
        # Asumimos que tu modelo acepta (x, edge_index, edge_attr, batch)
        predicciones_perturbadas = predict_in_batches(model, muestras_procesadas, device, batch_size=32)    
    # Asegurar la forma [num_samples, 1]
    predicciones_perturbadas = predicciones_perturbadas.view(-1, 1)

    # Convertir a tensor [num_samples,1]
    predicciones_perturbadas = torch.tensor(predicciones_perturbadas, dtype=torch.float, device=device).unsqueeze(1)

    # Calcular distancias entre la muestra original y las perturbaciones ( one hot)
    feature_distances = graph_feature_distance_list(muestra, perturbed_samples)

    # Obtener E (onehot)
    E_list = [data_z.x.to(device) for data_z in perturbed_samples]

    # Variable para activar/desactivar el filtro (cámbialo a False para comparar)
    aplicar_filtro_columnas = True
    aplicar_filtro_filas = True
    THRESHOLD = 1

    # --- NUEVO PASO: Calcular y filtrar proporcionalidades ---

    if aplicar_filtro_columnas:
        E_list, features_mantenidas, info_eliminada, num_feat_orig = calcular_y_filtrar_proporcionalidad(
            muestra.to(device),
            E_list,
            threshold=THRESHOLD,
        )
    else:
        info_eliminada = None
        num_feat_orig = E_list[0].shape[1] if len(E_list) > 0 else 0
        features_mantenidas = list(range(num_feat_orig)) # <--- IMPORTANTE AÑADIR ESTO
        print("[i] Filtro de proporcionalidad DESACTIVADO.")

    if aplicar_filtro_filas:
        # Nota: Le pasamos el E_list que ya viene limpio de columnas
        E_list, nodos_mantenidos, info_row, num_nodos_orig = calcular_y_filtrar_proporcionalidad(
            muestra.to(device), E_list, threshold=THRESHOLD, axis=1
        )



    # Lo mismo con los edges (onehot)
    A_list = []
    for data_z in perturbed_samples:
        if data_z.edge_attr is not None:
            A_list.append(data_z.edge_attr.to(device))
        else:
            print("Error al añadir los edge features de una molecula")

    if aplicar_filtro_columnas:
                A_list, features_mantenidas_e, info_eliminada_e, num_feat_orig_e = calcular_y_filtrar_proporcionalidad(
                    muestra.to(device),
                    A_list,
                    threshold=THRESHOLD,
                    mode="Edges",
                )
    if aplicar_filtro_filas:
                A_list, edges_mantenidos, info_row_e, num_edges_orig = calcular_y_filtrar_proporcionalidad(
                    muestra.to(device), A_list, threshold=THRESHOLD, mode="Edges", axis=1
                )

    # --------------- HACERLO CON OPTIMIZACION DE PYTORCH ----------------------
    # Si el filtro está en False, 'alfa_reducido' será en realidad el tensor completo
    alfa_reducido, beta_reducido, gamma_reducida, delta_reducida, loss = obtener_argmin(feature_distances, predicciones_perturbadas, E_list, A_list)

    # Si info_eliminada es None, reconstruir_alfa simplemente devuelve alfa_reducido tal cual
    alfa = reconstruir_tensor_importancias(alfa_reducido, num_feat_orig, features_mantenidas, info_eliminada)
    gamma = reconstruir_tensor_importancias(gamma_reducida, num_feat_orig_e, features_mantenidas_e, info_eliminada_e)
    beta = reconstruir_tensor_importancias(beta_reducido, num_nodos_orig, nodos_mantenidos, info_row, axis=1)
    delta = reconstruir_tensor_importancias(delta_reducida, num_edges_orig, edges_mantenidos, info_row_e, axis=1)

    # Verificar que aprendimos algo distinto de cero
    print(f"Max Alfa: {alfa.max().item():.4f}, Min Alfa: {alfa.min().item():.4f}")
    print(f"Max Beta: {beta.max().item():.4f}, Min Beta: {beta.min().item():.4f}")

    # ==========================================================================
    # PROCESAMIENTO DE MATRICES
    # ==========================================================================

    # 1. ALFA (Node Features) -> Filtrar -> Ordenar -> Normalizar
    node_feature_names = NODE_FEATURE_NAMES_ONE_HOT
    alfa_sorted, row_labels_alfa = procesar_features_onehot(
        alfa, node_feature_names, muestra.x
    )

    # 2. GAMMA (Edge Features) -> Filtrar -> Ordenar -> Normalizar
    # Reemplaza a Beta en el segundo heatmap
    if muestra.edge_attr is not None:
        # edge_feature_names = ["Bond Type", "Distance"]
        edge_feature_names = EDGE_FEATURE_NAMES
        
        gamma_sorted, row_labels_gamma = procesar_features_onehot(
            gamma, edge_feature_names, muestra.edge_attr
        )
    else:
        gamma_sorted = np.array([])
        row_labels_gamma = []

    # --- BETA (Nodos) ---
    beta_np = tensor_to_abs_numpy(beta)
    beta_np = normalizar_por_l2(beta_np)  

    # --- DELTA (Aristas) ---
    if delta is not None:
        delta_np = tensor_to_abs_numpy(delta)
        print("Delta crudo:", delta_np)
        # CAMBIO: Usar normalizar_por_l2 para evitar que una arista desaparezca si hay pocas
        delta_normalized = normalizar_por_l2(delta_np) 
    else:
        delta_normalized = np.array([])
    
    # NUEVA LÓGICA: Si es batch, preparamos el diccionario y retornamos INMEDIATAMENTE
    if batch_mode:
        return {
            'mol_name': mol_name, # Para usarlo como llave en el bucle
            'alfa': alfa_sorted if alfa is not None else None,
            'beta': beta_np if beta is not None else None,
            'gamma': gamma_sorted if gamma is not None else None,
            'delta': delta_normalized if delta is not None else None
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

def predict_in_batches(model, batch_data_list, device, batch_size=32):
    """Procesa predicciones en mini-batches para ahorrar memoria GPU"""
    all_predictions = []
    
    for i in range(0, len(batch_data_list), batch_size):
        mini_batch = Batch.from_data_list(batch_data_list[i:i+batch_size])
        mini_batch = mini_batch.to(device)
        
        with torch.no_grad():
            preds = model(mini_batch.x, mini_batch.edge_index, mini_batch.edge_attr, mini_batch.batch)
            all_predictions.append(preds)
        
        torch.cuda.empty_cache()  # Liberar memoria después de cada mini-batch
    
    return torch.cat(all_predictions, dim=0)