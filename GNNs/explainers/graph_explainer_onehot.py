# Graph explainer pero utilizando las features como one hot en vez de las del embedding

from GNNs.model_tester import cargar_modelo, predecir_molecula
import torch
import torch.nn as nn
import numpy as np
from ui.utils.constants import periodic_elements, hybridization_types, EDGE_FEATURE_NAMES
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
    normalizar_por_norma, get_features_names_onehot, 
    procesar_features_onehot )

ALGO_NAME = "GraphExplainer"
# Un 15% - 20% es razonable para mantener la estructura general.
MININICIAL = sys.float_info.max
logger = logging.getLogger(__name__)

# Función para dada una muestra (x), genere una muestra perturbada (Z)
# Dado un vector binario de las características a perturbar
def perturb_features_sample(data, feature_mask=[1, 1, 1, 1, 1, 1], noise_level=0.05, perturb_prob=0.5):
    data_new = data.clone()
    x = data_new.x
    num_nodes = x.shape[0]
    device = x.device
    
    # === 0. ANÁLISIS DE SPARSITY ===
    active_x_cols = (x != 0).any(dim=0)
    
    # === DEFINICIÓN DE INDICES ===
    len_atom = len(periodic_elements)
    len_hybrid = len(hybridization_types)
    
    start_atom, end_atom = 0, len_atom
    start_hybrid = x.shape[1] - len_hybrid
    end_hybrid = x.shape[1]
    
    if len(feature_mask) < 6:
        feature_mask = list(feature_mask) + [False] * (6 - len(feature_mask))

    # ==========================================
    # 1. PERTURBACIÓN DE NODOS (100% Vectorizado)
    # ==========================================
    # Tiramos TODAS las monedas a la vez: crea un tensor [num_nodes, 1] de booleanos
    monedas_nodos = (torch.rand(num_nodes, 1, device=device) < perturb_prob)

    # A. TIPO DE ÁTOMO (One-Hot Masking)
    if feature_mask[0]:
        # ~monedas_nodos invierte True a False. Al pasarlo a float:
        # Si tocó perturbar (True -> False -> 0.0), multiplicamos por 0 (lo apaga)
        # Si no (False -> True -> 1.0), multiplicamos por 1 (lo deja igual)
        x[:, start_atom:end_atom] *= (~monedas_nodos).float()

    # B. FEATURES CONTINUAS (Ruido)
    if feature_mask[1]:
        indices_continuous = [0, 1, 3, 4] 
        indices_absolutos = [idx + end_atom for idx in indices_continuous]
        indices_validos_abs = [idx for idx in indices_absolutos if active_x_cols[idx]]
        
        if len(indices_validos_abs) > 0:
            # Generamos todo el ruido a la vez
            ruido = noise_level * torch.randn(num_nodes, len(indices_validos_abs), device=device)
            # Solo aplicamos el ruido donde la moneda cayó en True
            ruido_aplicado = ruido * monedas_nodos.float()
            
            x[:, indices_validos_abs] += ruido_aplicado
            x[:, indices_validos_abs] = torch.clamp(x[:, indices_validos_abs], min=0.0, max=1.0)

    # C. FEATURES BINARIAS (Flip 1 -> 0 only)
    if feature_mask[2]:
        indices_binary = [2, 5, 6] 
        indices_bin_abs = [idx + end_atom for idx in indices_binary]
        
        # Máscara booleana: ¿Toca perturbar? Y ¿el valor es > 0.5?
        # Expandimos la moneda para que encaje con el número de columnas binarias
        condicion_flip = monedas_nodos & (x[:, indices_bin_abs] > 0.5)
        
        # PyTorch permite asignar valores directamente usando máscaras booleanas al instante
        x_bin_view = x[:, indices_bin_abs]
        x_bin_view[condicion_flip] = 0.0
        x[:, indices_bin_abs] = x_bin_view

    # D. HIBRIDACIÓN (One-Hot Masking)
    if feature_mask[3]:
        x[:, start_hybrid:end_hybrid] *= (~monedas_nodos).float()

    data_new.x = x

    # ==========================================
    # 2. PERTURBACIÓN DE ARISTAS (Simetría Inteligente)
    # ==========================================
    if data_new.edge_attr is not None:
        edge_attr = data_new.edge_attr
        edge_index = data_new.edge_index
        num_edges = edge_attr.shape[0]
        
        dist_idx = -1 
        num_bond_cols = edge_attr.shape[1] - 1 
        active_e_cols = (edge_attr != 0).any(dim=0)

        # Para vectorizar aristas y no romper la simetría (u,v == v,u), 
        # aislamos las aristas únicas (solo donde nodo_origen < nodo_destino)
        mask_unique = edge_index[0] < edge_index[1]
        indices_unique = torch.where(mask_unique)[0]
        num_unique = indices_unique.shape[0]

        # Tiramos monedas solo para las aristas maestras
        monedas_edges = (torch.rand(num_unique, 1, device=device) < perturb_prob)

        # Perturbar Tipo Enlace
        if feature_mask[4]:
            edge_attr[indices_unique, :num_bond_cols] *= (~monedas_edges).float()

        # Perturbar Distancia
        dist_idx_abs = edge_attr.shape[1] - 1
        if feature_mask[5] and active_e_cols[dist_idx_abs]:
            ruido_e = noise_level * torch.randn(num_unique, 1, device=device)
            ruido_e_aplicado = ruido_e * monedas_edges.float()
            
            # Usamos dist_idx:dist_idx+1 para mantener el slicing en 2D
            edge_attr[indices_unique, dist_idx:dist_idx+1] += ruido_e_aplicado
            edge_attr[indices_unique, dist_idx:dist_idx+1] = torch.clamp(edge_attr[indices_unique, dist_idx:dist_idx+1], min=0.0)

        # -- REPLICAR SIMETRÍA RÁPIDAMENTE --
        edge_map = {(edge_index[0, k].item(), edge_index[1, k].item()): k for k in range(num_edges)}
        
        # Copiamos la arista perturbada a su arista espejo
        for idx in indices_unique.tolist():
            u, v = edge_index[0, idx].item(), edge_index[1, idx].item()
            if (v, u) in edge_map:
                sym_idx = edge_map[(v, u)]
                edge_attr[sym_idx] = edge_attr[idx]

        data_new.edge_attr = edge_attr

    return data_new

# Función para generar múltiples muestras perturbadas
def generate_perturbed_samples(data, feature_mask, num_samples=50, noise_level=0.05):
    perturbed_samples = []
    for i in range(num_samples):
        sample_specific_prob = random.uniform(0.05, 0.95)
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
        num_samples=50, 
        noise_level=0.05, 
        device='cpu',
        batch_mode = False):
    
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    muestra = mol_to_graph_data(mol, 'one_hot')

    # Obtener nombre limpio de la molécula (ID) para buscar en el txt
    mol_id = os.path.basename(sdf_path).split('.')[0]
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else mol_id

    # --- 1. OBTENER INFORMACIÓN REAL ---
    target_name_str, real_val = obtener_info_real(target_data_path, mol_id)
    print(real_val)
    
    # Generar muestras perturbadas
    perturbed_samples = generate_perturbed_samples(muestra, feature_mask, num_samples, noise_level)
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

    # Lo mismo con los edges (onehot)
    A_list = []
    for data_z in perturbed_samples:
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

    # NUEVA LÓGICA: Si es batch, preparamos el diccionario y retornamos INMEDIATAMENTE
    if batch_mode:
        return {
            'mol_name': mol_name, # Para usarlo como llave en el bucle
            'alfa': alfa.detach().cpu() if alfa is not None else None,
            'beta': beta.detach().cpu() if beta is not None else None,
            'gamma': gamma.detach().cpu() if gamma is not None else None,
            'delta': delta.detach().cpu() if delta is not None else None
        }

    guardar_pesos(alfa, beta, gamma, delta, model_folder_name,
                  mol_name, ALGO_NAME)
    
    # ==========================================================================
    # PROCESAMIENTO DE MATRICES
    # ==========================================================================
    
    # 1. ALFA (Node Features) -> Filtrar -> Ordenar -> Normalizar
    node_feature_names = get_features_names_onehot()
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
    l1_lambda = 1e-4  # Antes quizás era muy alto comparado con el gradiente

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