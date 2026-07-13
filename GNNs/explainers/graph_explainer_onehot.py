# Graph explainer pero utilizando las features como one hot en vez de las del embedding

from GNNs.model_tester import cargar_modelo, predecir_molecula
import torch
import torch.nn as nn
import numpy as np
import os
from torch_geometric.data import Batch
import sys
import logging
from GNNs.data_processing import mol_to_graph_data, onehot_to_indices
from rdkit import Chem
import random
from graph_managment.sdf_converter import parse_sdf
from GNNs.explainers.explanation_helper import ( 
    obtener_info_real, guardar_dashboard_explicacion,
    guardar_pesos, tensor_to_abs_numpy, 
    normalizar_por_l2, 
    procesar_features_onehot, get_features_names_onehot )
from ui.utils.constants import (
    EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES_ONE_HOT,
    ONE_HOT_INDICES, EDGE_ONE_HOT_INDICES)

ALGO_NAME = "GraphExplainer"
FEATURE_MASK_LENGTH = 8
# Un 15% - 20% es razonable para mantener la estructura general.
MININICIAL = sys.float_info.max
logger = logging.getLogger(__name__)

# Función para dada una muestra (x), genere una muestra perturbada (Z)
# Dado un vector binario de las características a perturbar
def perturb_features_sample(data, feature_mask=[1, 1, 1, 1, 1, 1, 1, 1], noise_level=0.05, perturb_prob=0.2):
    # feature_mask ahora tiene longitud 8:
    # 0: Node Atom Type
    # 1: Node Continuous
    # 2: Node Binary
    # 3: Node Hybridization
    # 4: Edge Covalent Bond Type
    # 5: Edge Distance
    # 6: Edge Flexibility
    # 7: Edge Non-Covalent Interactions (NUEVO)
    
    data_new = data.clone()
    x = data_new.x
    num_nodes = x.shape[0]
    device = x.device
    
    # === 0. ANÁLISIS DE SPARSITY ===
    active_x_cols = (x != 0).any(dim=0)
    
    # Rellenar máscara si es más corta
    if len(feature_mask) < FEATURE_MASK_LENGTH:
        feature_mask = list(feature_mask) + [False] * (FEATURE_MASK_LENGTH - len(feature_mask))

    # ==========================================
    # 1. PERTURBACIÓN DE NODOS (100% Vectorizado)
    # ==========================================
    for i in range(num_nodes):
        
        # A. TIPO DE ÁTOMO (One-Hot)
        if feature_mask[0] and torch.rand(1).item() < perturb_prob:
            atom_slice = ONE_HOT_INDICES["ATOM_SYMBOL"]
            onehot = x[i, atom_slice]
            if onehot.nonzero(as_tuple=False).numel() > 0:
                x[i, atom_slice] = 0 

        # B. FEATURES CONTINUAS (Ruido)
        if feature_mask[1] and torch.rand(1).item() < perturb_prob:
            indices_continuous = [ONE_HOT_INDICES["DEGREE"], ONE_HOT_INDICES["TOTAL_HS"]]
            indices_validos = [idx for idx in indices_continuous if active_x_cols[idx]]
            
            if indices_validos:
                noise = noise_level * torch.randn(len(indices_validos)).to(device)
                x[i, indices_validos] += noise
                x[i, indices_validos] = torch.clamp(x[i, indices_validos], min=0.0, max=1.0)

        # C. FEATURES BINARIAS (Flip 1 -> 0)
        if feature_mask[2]:
            indices_binary = [ONE_HOT_INDICES["IS_AROMATIC"], ONE_HOT_INDICES["IS_DONOR"], ONE_HOT_INDICES["IS_ACCEPTOR"]]
            for idx in indices_binary:
                if torch.rand(1).item() < perturb_prob and x[i, idx] > 0.5:
                    x[i, idx] = 0.0

        # D. HIBRIDACIÓN (One-Hot)
        if feature_mask[3] and torch.rand(1).item() < perturb_prob:
            hyb_slice = ONE_HOT_INDICES["HYBRIDIZATION"]
            onehot = x[i, hyb_slice]
            if (onehot == 1).any():
                x[i, hyb_slice] = 0

    data_new.x = x

    # ==========================================
    # 2. PERTURBACIÓN DE ARISTAS (Simetría Inteligente)
    # ==========================================
    if data_new.edge_attr is not None:
        edge_attr = data_new.edge_attr
        edge_index = data_new.edge_index
        num_edges = edge_attr.shape[0]
        
        # Mapa para mantener simetría en grafos no dirigidos
        edge_map = {(edge_index[0, k].item(), edge_index[1, k].item()): k for k in range(num_edges)}
        
        # Chequeo de columnas activas en aristas
        active_e_cols = (edge_attr != 0).any(dim=0)

        for i in range(num_edges):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            # Iteramos solo en una dirección, luego copiamos al reverso
            if u > v: 
                continue 

            modified = False
            
            # A. Perturbar Tipo Enlace Covalente -> feature_mask[4]
            if feature_mask[4] and torch.rand(1).item() < perturb_prob:
                bond_slice = EDGE_ONE_HOT_INDICES["BOND_TYPE"]
                onehot_bond = edge_attr[i, bond_slice]
                if onehot_bond.nonzero(as_tuple=False).numel() > 0:
                    edge_attr[i, bond_slice] = 0.0
                    modified = True
            
            # B. Perturbar Distancia -> feature_mask[5]
            dist_idx = EDGE_ONE_HOT_INDICES["DISTANCE"]
            if feature_mask[5] and active_e_cols[dist_idx]:
                if torch.rand(1).item() < perturb_prob: 
                    noise = noise_level * torch.randn(1).item()
                    edge_attr[i, dist_idx] += noise
                    edge_attr[i, dist_idx] = max(0.0, edge_attr[i, dist_idx].item())
                    modified = True

            # C. Perturbar Flexibilidad -> feature_mask[6]
            flex_idx = EDGE_ONE_HOT_INDICES["FLEXIBILITY"]
            if feature_mask[6] and active_e_cols[flex_idx]:
                if torch.rand(1).item() < perturb_prob and edge_attr[i, flex_idx] > 0.5:
                    edge_attr[i, flex_idx] = 0.0
                    modified = True
                    
            # D. Perturbar Interacciones No Covalentes (NUEVO) -> feature_mask[7]
            if feature_mask[7]:
                nc_slice = EDGE_ONE_HOT_INDICES["NON_COVALENT"]
                nc_vector = edge_attr[i, nc_slice]
                
                # Si hay alguna interacción activa (el vector no son puros 0s)
                if nc_vector.sum() > 0:
                    for k in range(nc_vector.shape[0]):
                        # Probabilidad independiente de 'borrar' cada interacción (flip de 1.0 a 0.0)
                        if nc_vector[k] > 0.5 and torch.rand(1).item() < perturb_prob:
                            # nc_slice.start nos da el índice base donde empieza el vector en el tensor
                            abs_idx = nc_slice.start + k
                            edge_attr[i, abs_idx] = 0.0
                            modified = True

            # Mantener simetría (u,v) == (v,u)
            if modified and (v, u) in edge_map:
                sym_idx = edge_map[(v, u)]
                # ¡CORRECCIÓN AQUÍ! Antes tenías edge_attr[idx] que no existía.
                edge_attr[sym_idx] = edge_attr[i].clone()

        data_new.edge_attr = edge_attr

    return data_new

# Función para generar múltiples muestras perturbadas
def generate_perturbed_samples(data, feature_mask=[1, 1, 1, 1, 1, 1, 1, 1], num_samples=50, noise_level=0.05):
    perturbed_samples = []
    for i in range(num_samples):
        sample_specific_prob = random.uniform(0.01, 0.50)
        # sample_specific_prob = random.uniform(0.01, 0.99)

        perturbed_sample = perturb_features_sample(data, feature_mask, noise_level, sample_specific_prob)
        perturbed_samples.append(perturbed_sample)
    return perturbed_samples

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
        num_samples=1000, 
        noise_level=0.01,
        batch_mode = False):
    
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    muestra = mol_to_graph_data(mol, 'one_hot')

    # Obtener nombre limpio de la molécula (ID) para buscar en el txt
    mol_id = os.path.basename(sdf_path).split('.')[0]
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else mol_id

    # --- 1. OBTENER INFORMACIÓN REAL ---
    target_name_str, real_val = obtener_info_real(target_data_path, mol_id)
    print("Real value:", real_val)
    
    # Generar muestras perturbadas
    perturbed_samples = generate_perturbed_samples(muestra, num_samples, noise_level)
    # perturbed_samples_embedding = []

    # Obtener modelo
    model, device, target_name = cargar_modelo(checkpoint_path)

    muestra_for_model = onehot_to_indices(muestra.to(device))
    prediccion_original = predecir_molecula(model, muestra_for_model, device)

    # 1. Convertir todas las muestras al formato del modelo en una lista
    muestras_procesadas = [onehot_to_indices(p.to(device)) for p in perturbed_samples]
    
    # 2. Unirlas en un solo "Súper Grafo" (Batch)
    batch_data = Batch.from_data_list(muestras_procesadas)
    
    # 3. Hacer UNA SOLA predicción masiva
    with torch.no_grad():
        # Asumimos que tu modelo acepta (x, edge_index, edge_attr, batch)
        predicciones_perturbadas = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
    
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
    alfa = reconstruir_importancias(alfa_reducido, num_feat_orig, features_mantenidas, info_eliminada)
    gamma = reconstruir_importancias(gamma_reducida, num_feat_orig_e, features_mantenidas_e, info_eliminada_e)
    beta = reconstruir_importancias(beta_reducido, num_nodos_orig, nodos_mantenidos, info_row)
    delta = reconstruir_importancias(delta_reducida, num_edges_orig, edges_mantenidos, info_row_e)

    # Verificar que aprendimos algo distinto de cero
    print(f"Max Alfa: {alfa.max().item():.4f}, Min Alfa: {alfa.min().item():.4f}")
    print(f"Max Beta: {beta.max().item():.4f}, Min Beta: {beta.min().item():.4f}")

    # Preparamos el nombre del modelo para la carpeta
    model_folder_name = checkpoint_path.split('/')[-1].split('.')[0]

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

    guardar_pesos(alfa, beta, gamma, delta, model_folder_name,
                  mol_name, ALGO_NAME)

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

    return plotfilename

def obtener_argmin(feature_distances, predicciones_perturbadas, 
                   E_list, A_list,
                   sigma = 1, 
                   lr=0.01, 
                   epochs=2000, 
                   verbose=True):
    
    device = predicciones_perturbadas.device
    
    # --- DIAGNÓSTICO DE VARIANZA ---
    # Si esto es 0 o muy bajo, no puede aprender nada porque el modelo
    # predice lo mismo para todas las perturbaciones.
    std_preds = predicciones_perturbadas.std().item()
    if verbose:
        logger.info(f"Desviación estándar de las predicciones: {std_preds:.6f}")
        if std_preds < 1e-5:
            logger.warning("¡CUIDADO! El modelo predice casi lo mismo para todas las muestras. Aumenta el noise_level.")

    # 1. Stack y normalizar
    E_stack = stack_and_normalize(E_list, device)
    has_edges = len(A_list) > 0 and A_list[0] is not None
    if has_edges:
        A_stack = stack_and_normalize(A_list, device)
        M_edges, d_edges = A_stack.shape[1], A_stack.shape[2]
    else:
        d_edges, M_edges = 1, 1

    num_samples, N_nodes, d_nodes = E_stack.shape 
    # N_nodes: Cantidad de átomos
    # d_features: Cantidad de features

    # Normalizar nodos
    scale_nodes = 1.0 / d_nodes * N_nodes
    
    # Para Edges
    if has_edges:
        # Evitamos división por cero si N_edges es muy pequeño
        denom_edges = d_edges * M_edges
        scale_edges = 1.0 / denom_edges if denom_edges > 0 else 1.0
    else:
        scale_edges = 0.0

    # 3. Inicialización (Un poco más grande para ayudar al gradiente)
    # Son los dos tensores columna
    alfa = nn.Parameter(torch.randn(d_nodes, 1, device=device) * 0.1)
    beta = nn.Parameter(torch.randn(N_nodes, 1, device=device) * 0.1)
    
    mean_pred = predicciones_perturbadas.mean().item()
    mu = nn.Parameter(torch.tensor([mean_pred], device=device, dtype=torch.float))
    # mu = nn.Parameter(torch.tensor([0.0], device=device, dtype=torch.float))

    params = [alfa, beta, mu]
    # params = [alfa, beta]
    
    gamma = None
    delta = None

    if has_edges:
        gamma = nn.Parameter(torch.randn(d_edges, 1, device=device) * 0.1)
        delta = nn.Parameter(torch.randn(M_edges, 1, device=device) * 0.1)
        params.extend([gamma, delta])
    
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

    # Todas las distancias pasadas a tensor
    dists = torch.tensor(feature_distances, dtype=torch.float, device=device).view(-1, 1)
    
    # weights = e^-(distancias²) / sigma²
    weights = torch.exp(-(dists**2) / sigma**2) # / (2 * sigma**2))
    
    # Se reescala weights para que sumen "num_samples"
    # Como esto multiplica el loss, si es muy pequeño, deja de optimizar
    # Esto evita que el Loss sea pequeñísimo (0.0001) y que los gradientes mueran.
    weights = (weights / weights.sum()) * num_samples

    targets = predicciones_perturbadas.view(-1, 1)

    # Bajamos un poco más el learning rate si ahora los gradientes son fuertes
    # O lo dejamos igual, pero vigilando.
    
    # ... (resto del código sigue igual)

    # --- CAMBIO CLAVE: MENOS REGULARIZACIÓN INICIAL ---
    l1_lambda = 0.01  # Antes quizás era muy alto comparado con el gradiente

    # MEJORA 1: Usar AdamW en lugar de Adam estándar
    # weight_decay ayuda a mantener los pesos controlados sin ser tan agresivo como L1
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    
    # MEJORA 2: Scheduler tipo Coseno
    # Esto baja el LR suavemente desde 0.05 hasta 0 al final de las epochs.
    # Evita que se estanque tan pronto.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Variables para calcular R^2 al final
    initial_loss = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- Nodos ---
        # Eb = torch.matmul(E_stack, beta)
        # hacemos E * alfa^t
        Ea = torch.matmul(E_stack, alfa)
        # Luego B^t * E * alfa^t
        # term_nodes = torch.matmul(alfa.t(), Eb).view(-1, 1)
        term_nodes = (Ea * beta).sum(dim=1)
        
        pred_approx = (term_nodes * scale_nodes) + mu
        # pred_approx = term_nodes * scale_nodes

        # --- Edges ---
        if has_edges:
            # Hacemos A * gamma^t
            Ag = torch.matmul(A_stack, gamma)
            # Ad = torch.matmul(A_stack, delta)
            # term_edges = torch.matmul(gamma.t(), Ad).view(-1, 1)
            # y ahora delta^t * A * gamma^t
            term_edges = (Ag * delta).sum(dim=1)

            pred_approx += (term_edges * scale_edges)

        # LOSS
        squared_error = (targets - pred_approx)**2
        loss = (weights * squared_error).mean() # Usar mean ayuda a estabilizar respecto al batch size
        
        # Guardamos el loss inicial real (sin regularización) para comparar
        if epoch == 0:
            initial_loss = loss.item()

        l1_reg = torch.norm(beta, 1) + torch.norm(alfa, 1) 
        if has_edges:
             l1_reg += torch.norm(delta, 1) + torch.norm(gamma, 1)
             
        total_loss = loss + (l1_lambda * l1_reg)

        total_loss.backward()
        optimizer.step()
        
        loss_val = total_loss.item()
        scheduler.step()
        
        if verbose and epoch % 200 == 0:
             # Imprimimos también el bias para ver si se mueve
             logger.info(f"Epoch {epoch}: Loss {loss_val:.5f} | Mu: {mu.item():.3f}")
    
    # --- MEJORA 3: CÁLCULO DE R^2 FINAL ---
    # R^2 = 1 - (Loss Final / Loss Inicial)
    # Esto te dice el % de la varianza explicada.
    r_squared = 1.0 - (loss_val / (initial_loss + 1e-8))
    logger.info(f"--- Entrenamiento Finalizado ---")
    logger.info(f"R² (Varianza Explicada): {r_squared:.2%} (Ideal > 80%)")

    return alfa.detach(), beta.detach(), (gamma.detach() if has_edges else None), (delta.detach() if has_edges else None), loss_val

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

def calcular_y_filtrar_proporcionalidad(original, element_list, threshold=0.85, mode="Nodos", axis=0):
    """
    Filtra múltiples colinealidades y ceros.
    axis=0 : Analiza Columnas (Features)
    axis=1 : Analiza Filas (Nodos / Aristas)
    """
    if mode == "Nodos":
        X = original.x
        feature_names = get_features_names_onehot()
    else:
        X = original.edge_attr
        feature_names = EDGE_FEATURE_NAMES

    working_X = X if axis == 0 else X.T
    num_elements = working_X.shape[1]

    if axis == 0:
        names = feature_names
        axis_name = "Columnas/Features"
    else:
        names = [f"Fila_{i}" for i in range(num_elements)]
        axis_name = "Filas/Elementos"

    # --- 1. Detectar y purgar puros ceros ---
    zero_mask = ~(working_X != 0).any(dim=0)
    zero_idx = zero_mask.nonzero(as_tuple=True)[0].tolist()

    if len(zero_idx) > 0:
        print(f"[i] Se detectaron {len(zero_idx)} {axis_name} de puros ceros en {mode}. Se omitirán.")

    # --- 2. Cálculos de matrices ---
    dot_products = torch.matmul(working_X.T, working_X)
    norms = torch.norm(working_X, dim=0)
    norms_matrix = torch.outer(norms, norms) + 1e-8
    
    cos_theta = dot_products / norms_matrix
    dop_matrix = torch.abs(cos_theta)

    norms_sq = torch.pow(norms, 2) + 1e-8
    p_matrix = dot_products / norms_sq.view(1, -1) 
    
    # Ignorar la diagonal
    dop_matrix.fill_diagonal_(0)

    # --- 3. Buscar TODAS las dependencias ---
    # Usamos la matriz triangular superior para no evaluar el par (A,B) y (B,A)
    upper_tri = torch.triu(dop_matrix, diagonal=1)
    indices_sobre_umbral = torch.nonzero(upper_tri >= threshold)
    
    pares_dependencia = []
    for idx in indices_sobre_umbral:
        j, k = idx[0].item(), idx[1].item()
        dop_val = dop_matrix[j, k].item()
        p_val = p_matrix[j, k].item()
        pares_dependencia.append((dop_val, j, k, p_val))

    # Ordenar por el DoP más alto primero (para resolver las correlaciones más fuertes antes)
    pares_dependencia.sort(key=lambda x: x[0], reverse=True)

    # --- 4. Iterar y establecer dependencias ---
    elements_to_keep = set(range(num_elements))
    for z in zero_idx:
        elements_to_keep.discard(z)

    # Ahora guardamos una LISTA de diccionarios
    info_eliminada_list = []

    for dop_val, j, k, p_val in pares_dependencia:
        # Solo establecemos dependencia si AMBOS elementos siguen vivos.
        # Si 'j' ya fue eliminado por otra variable, lo saltamos.
        # Si 'k' ya fue eliminado, no lo podemos usar de base, lo saltamos.
        if j in elements_to_keep and k in elements_to_keep:
            elements_to_keep.remove(j)
            info_eliminada_list.append({
                'j': j,
                'k': k,
                'p': p_val,
            })
            
            nombre_j = names[j]
            nombre_k = names[k]
            print(f"[!] Eliminando '{nombre_j}' (DoP: {dop_val:.4f}). Será función de '{nombre_k}'.")

    elements_to_keep = sorted(list(elements_to_keep))
    elementos_eliminados = num_elements - len(elements_to_keep) - len(zero_idx)
    
    if elementos_eliminados > 0:
        print(f"\n--- Resumen de Proporcionalidad ({mode} - {axis_name}) ---")
        print(f"Total eliminados por colinealidad: {elementos_eliminados}")
        print("----------------------------------------------------------\n")
    else:
        print(f"[i] Ningún DoP supera el threshold de {threshold} en {mode} ({axis_name}).")

    # --- 5. Filtrado Final ---
    if axis == 0:
        element_list_filtrado = [data[:, elements_to_keep] for data in element_list]
    else:
        element_list_filtrado = [data[elements_to_keep, :] for data in element_list]

    return element_list_filtrado, elements_to_keep, info_eliminada_list, num_elements

def reconstruir_importancias(tensor_reducido, dimension_original, indices_mantenidos, info_eliminada_list):
    """
    Expande un tensor de pesos a su tamaño original, rellenando con 0s y 
    aplicando la Ecuación 16 en ORDEN INVERSO para resolver cadenas de dependencias.
    """
    tensor_completo = torch.zeros((dimension_original, 1), device=tensor_reducido.device)
    
    # 1. Colocar las variables independientes optimizadas
    for idx_reducido, idx_original in enumerate(indices_mantenidos):
        tensor_completo[idx_original] = tensor_reducido[idx_reducido]
        
    # 2. Reconstruir dependencias desde el final hacia el principio
    if info_eliminada_list:
        for info in reversed(info_eliminada_list):
            j = info['j']
            k = info['k']
            p = info['p']
            
            # El valor 'k' ya debe estar reconstruido gracias al orden inverso
            tilde_k = tensor_completo[k].clone()
            
            tensor_completo[k] = (1 + p) * tilde_k - p
            tensor_completo[j] = 1.0
            
    return tensor_completo