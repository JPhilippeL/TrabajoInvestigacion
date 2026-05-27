# model_explainer.py
from GNNs.model_tester import cargar_modelo, predecir_molecula
import torch
import torch.nn as nn
import numpy as np
from ui.utils.constants import periodic_elements, hybridization_types
import os
import sys
import logging
from GNNs.data_processing import mol_to_graph_data, onehot_to_indices
from rdkit import Chem
from graph_managment.sdf_converter import parse_sdf
from GNNs.explainers.explanation_helper import ( 
    obtener_info_real, guardar_dashboard_explicacion,
    guardar_pesos, tensor_to_abs_numpy, 
    normalizar_por_norma, get_feature_names_embedding, 
    procesar_features_ordenadas )
from GNNs.explainers.explanation_fidelity import calcular_curvas_fidelity_general, guardar_plot_fidelity
from GNNs.explainers.graph_explainer_onehot import obtener_argmin

ALGO_NAME = "GraphExplainer"
# Probabilidad de que un nodo/arista específico sea modificado.
PERTURB_PROB = 0.15
# Un 15% - 20% es razonable para mantener la estructura general.
MININICIAL = sys.float_info.max
logger = logging.getLogger(__name__)

# Función para dada una muestra (x), genere una muestra perturbada (Z)
# Dado un vector binario de las características a perturbar
def perturb_features_sample(data, feature_mask=[1, 1, 1, 1, 1, 1], noise_level=0.05):
    data_new = data.clone()
    x = data_new.x
    num_nodes = x.shape[0]
    
    # === DEFINICIÓN DE INDICES ===
    len_atom = len(periodic_elements)
    len_hybrid = len(hybridization_types)
    
    start_atom, end_atom = 0, len_atom
    start_hybrid = x.shape[1] - len_hybrid
    end_hybrid = x.shape[1]
    
    # Rellenar máscara
    if len(feature_mask) < 6:
        feature_mask = list(feature_mask) + [False] * (6 - len(feature_mask))

    # ==========================================
    # 1. PERTURBACIÓN DE NODOS
    # ==========================================
    for i in range(num_nodes):
        
        # A. TIPO DE ÁTOMO (One-Hot) - SOLO si toca la lotería (rand < PERTURB_PROB)
        if feature_mask[0] and torch.rand(1).item() < PERTURB_PROB:
            onehot = x[i, start_atom:end_atom]
            onehot[:] = 0
            new_idx = torch.randint(0, len_atom, (1,))
            onehot[new_idx] = 1

        # B. FEATURES CONTINUAS (Ruido)
        # El ruido continuo SÍ se puede aplicar a todos (es suave), 
        # o también puedes hacerlo probabilístico. Aquí lo dejo a todos pero suave.
        # B. FEATURES CONTINUAS (Ruido)
        if feature_mask[1]:
            indices_continuous = [0, 1, 3, 4] # Índices relativos al slice
            vals = x[i, end_atom:start_hybrid]
            # Generar ruido para todos los índices de una vez
            noise = noise_level * torch.randn(len(indices_continuous))
            # Sumar ruido (vectorizado)
            vals[indices_continuous] += noise
            # Clamping (vectorizado) - Esto reemplaza tu bucle if/elif
            vals[indices_continuous] = torch.clamp(vals[indices_continuous], min=0.0, max=1.0)

        # C. FEATURES BINARIAS (Flip) - CRÍTICO: Hacerlo Probabilístico
        if feature_mask[2]:
            indices_binary = [2, 5, 6] # Aromatic, Donor, Acceptor
            vals = x[i, end_atom:start_hybrid]
            
            for idx_rel in indices_binary:
                # Solo invertimos el bit con probabilidad PERTURB_PROB
                if torch.rand(1).item() < PERTURB_PROB:
                    vals[idx_rel] = 1.0 - vals[idx_rel]

        # D. HIBRIDACIÓN (One-Hot) - Probabilístico
        if feature_mask[3] and torch.rand(1).item() < PERTURB_PROB:
            onehot = x[i, start_hybrid:end_hybrid]
            onehot[:] = 0
            new_idx = torch.randint(0, len_hybrid, (1,))
            onehot[new_idx] = 1

    data_new.x = x

    # ==========================================
    # 2. PERTURBACIÓN DE ARISTAS
    # ==========================================
    if data_new.edge_attr is not None:
        edge_attr = data_new.edge_attr
        edge_index = data_new.edge_index
        num_edges = edge_attr.shape[0]
        
        edge_map = {(edge_index[0, k].item(), edge_index[1, k].item()): k for k in range(num_edges)}
        dist_idx = -1 
        num_bond_cols = edge_attr.shape[1] - 1 

        for i in range(num_edges):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if u > v: continue 

            modified = False
            
            # Perturbar Tipo Enlace (Probabilístico)
            if feature_mask[4] and torch.rand(1).item() < PERTURB_PROB:
                onehot_bond = edge_attr[i, :num_bond_cols]
                onehot_bond[:] = 0
                new_bond_idx = torch.randint(0, num_bond_cols, (1,))
                onehot_bond[new_bond_idx] = 1
                modified = True
            
            # Perturbar Distancia (Siempre un poco de ruido está bien, o hazlo probabilístico)
            if feature_mask[5]: 
                noise = noise_level * torch.randn(1).item()
                edge_attr[i, dist_idx] += noise
                edge_attr[i, dist_idx] = torch.clamp(edge_attr[i, dist_idx], min=0.0)
                modified = True

            if modified and (v, u) in edge_map:
                sym_idx = edge_map[(v, u)]
                edge_attr[sym_idx] = edge_attr[i].clone()

        data_new.edge_attr = edge_attr

    return data_new

# Función para generar múltiples muestras perturbadas
def generate_perturbed_samples(data, feature_mask, num_samples=50, noise_level=0.05):
    perturbed_samples = []
    for i in range(num_samples):
        perturbed_sample = perturb_features_sample(data, feature_mask, noise_level)
        perturbed_samples.append(perturbed_sample)
    return perturbed_samples

import torch

def calculate_frobenius_distance(tensor_a, tensor_b):
    """Calcula la distancia Frobenius normalizada por el tamaño."""
    if tensor_a.shape != tensor_b.shape:
        # Tienen q ser iguales
        return torch.tensor(0.0, device=tensor_a.device) 
        
    diff = tensor_a - tensor_b
    # Norma Frobenius
    frob_dist = torch.norm(diff, p='fro')
    
    # Normalización propuesta por el profesor: (num_elementos * num_atributos)
    # Esto es numel() en PyTorch (total de elementos en la matriz)
    normalization_factor = tensor_a.numel() 
    
    if normalization_factor == 0: return torch.tensor(0.0, device=tensor_a.device)
    
    return frob_dist / normalization_factor

def graph_feature_distance_list(x, z_list, epsilon=0.5):
    """
    Calcula la distancia combinada (Nodos + Aristas) usando norma Frobenius.
    D = eps * dist_nodos + (1-eps) * dist_aristas
    """
    distances = []
    
    # Pre-calculamos factores de la muestra original
    x_nodes = x.x
    x_edges = x.edge_attr if x.edge_attr is not None else torch.empty(0)
    
    for z in z_list:
        z_nodes = z.x
        z_edges = z.edge_attr if z.edge_attr is not None else torch.empty(0)

        # 1. Distancia de Nodos
        dist_n = calculate_frobenius_distance(x_nodes, z_nodes)
        
        # 2. Distancia de Aristas
        if x_edges.numel() > 0 and z_edges.numel() > 0:
            dist_e = calculate_frobenius_distance(x_edges, z_edges)
        else:
            # Si no hay aristas en la molécula original, la distancia es 0
            dist_e = torch.tensor(0.0, device=x.device)

        # 3. Combinación Ponderada (Fórmula del profesor)
        # Epsilon controla cuánto pesan los nodos vs las aristas
        combined_dist = (epsilon * dist_n) + ((1 - epsilon) * dist_e)
        
        distances.append(combined_dist.item())
    
    return distances

def obtener_graph_explainer(
        checkpoint_path, 
        sdf_path, 
        target_data_path=None, 
        feature_mask=[1, 1, 1, 1, 1, 1], 
        num_samples=50, 
        noise_level=0.05, 
        device='cpu',
        imagen = True):
    
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    muestra = mol_to_graph_data(mol, 'one_hot')

    # Obtener nombre limpio de la molécula (ID) para buscar en el txt
    mol_id = os.path.basename(sdf_path).split('.')[0]
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else mol_id

    # --- 1. OBTENER INFORMACIÓN REAL ---
    target_name_str, real_val = obtener_info_real(target_data_path, mol_id)
    
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

    # Preparamos el nombre del modelo para la carpeta
    model_folder_name = checkpoint_path.split('/')[-1].split('.')[0]

    guardar_pesos(alfa, beta, gamma, delta, model_folder_name,
                  mol_name, ALGO_NAME)
    
    if imagen == False:
        logger.info("Pesos guardados, no se hizo imagen")
        return 1
    
    # ==========================================================================
    # PROCESAMIENTO DE MATRICES
    # ==========================================================================
    
    # 1. ALFA (Node Features) -> Filtrar -> Ordenar -> Normalizar
    node_feature_names = get_feature_names_embedding()
    alfa_sorted, row_labels_alfa = procesar_features_ordenadas(
        alfa, node_feature_names, muestra_for_model.x
    )

    # 2. GAMMA (Edge Features) -> Filtrar -> Ordenar -> Normalizar
    # Reemplaza a Beta en el segundo heatmap
    if muestra.edge_attr is not None:
        edge_feature_names = ["Bond Type", "Distance"]
        
        gamma_sorted, row_labels_gamma = procesar_features_ordenadas(
            gamma, edge_feature_names, muestra_for_model.edge_attr
        )
    else:
        gamma_sorted = np.array([])
        row_labels_gamma = []

    # --- BETA (Nodos) ---
    beta_np = tensor_to_abs_numpy(beta)
    # CAMBIO: Usar normalizar_por_norma para ser consistente con los heatmaps
    beta_np = normalizar_por_norma(beta_np)  

    # --- DELTA (Aristas) ---
    if delta is not None:
        delta_np = tensor_to_abs_numpy(delta)
        # CAMBIO: Usar normalizar_por_norma para evitar que una arista desaparezca si hay pocas
        delta_normalized = normalizar_por_norma(delta_np) 
    else:
        delta_normalized = np.array([])


    # LLAMADA A LA FUNCIÓN Visualizacion
    plotfilename = guardar_dashboard_explicacion(
        graph_obj=parse_sdf(sdf_path),
        edge_index=muestra_for_model.edge_index,
        node_importance=beta_np.flatten(),
        edge_importance=delta_normalized.flatten(),
        
        alfa_sorted=alfa_sorted,
        row_labels_alfa=row_labels_alfa,
        gamma_sorted=gamma_sorted,
        row_labels_gamma=row_labels_gamma,
        
        mol_name=mol_name,
        target_name=target_name_str,
        real_val=real_val,
        pred_val=prediccion_original,
        algo_name=ALGO_NAME,
        model_name=model_folder_name
    )

    # 1. Calcular Curvas de FIABILIDAD
    k_vals, fiab_minus = calcular_curvas_fidelity_general(
        model, 
        muestra_for_model, 
        beta.abs(), 
        device
    )

    # 3. Guardar (Solo pasamos datos puros)
    fiab_path = guardar_plot_fidelity(
        k_values=k_vals,
        fiab_minus=fiab_minus, 
        model_name=model_folder_name,
        mol_name=mol_name,
        algo_name=ALGO_NAME
    )
    
    logger.info(f"Gráfico fidelity guardado en: {fiab_path}")

    return plotfilename

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