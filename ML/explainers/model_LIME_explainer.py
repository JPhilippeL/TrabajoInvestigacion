# model_explainer.py
from ML.model_tester import cargar_modelo, predecir_molecula
import torch
import torch.nn as nn
import numpy as np
from ui.utils import RESULTADOS_DIR, periodic_elements, hybridization_types
import os
import sys
import logging
from ML.data_processing import mol_to_graph_data, onehot_to_indices
from rdkit import Chem
from core.sdf_converter import parse_sdf
from ML.explainers.explanation_visualization import obtener_info_real, guardar_dashboard_explicacion

SIGMADIST = 1
MININICIAL = sys.float_info.max
logger = logging.getLogger(__name__)

# Función para dada una muestra (x), genere una muestra perturbada (Z)
# Dado un vector binario de las características a perturbar
def perturb_features_sample(data, feature_mask=[1, 1, 1, 1, 1, 1], noise_level=0.05):
    data_new = data.clone()
    x = data_new.x
    num_nodes = x.shape[0]

    # Probabilidad de que un nodo/arista específico sea modificado.
    # Un 15% - 20% es razonable para mantener la estructura general.
    PERTURB_PROB = 0.15 
    
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
        if feature_mask[1]:
            indices_continuous = [0, 1, 3, 4] 
            vals = x[i, end_atom:start_hybrid]
            noise = noise_level * torch.randn(len(indices_continuous))
            for k, idx_rel in enumerate(indices_continuous):
                vals[idx_rel] += noise[k]

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

def graph_feature_distance_list(x, z_list, metric='euclidean'):

    # distancia promedio usando las features de nodos

    distances = []
    for z in z_list:
        if x.x.shape != z.x.shape:
            raise ValueError("x y z deben tener la misma forma de features")

        diff = x.x - z.x

        if metric == 'euclidean':
            dist = torch.norm(diff, dim=1).mean()
        elif metric == 'cosine':
            sim = torch.nn.functional.cosine_similarity(x.x, z.x, dim=1)
            dist = 1 - sim.mean()
        else:
            raise ValueError(f"Métrica '{metric}' no soportada.")
        
        distances.append(dist.item())
    
    return distances

def obtener_lime(
        checkpoint_path, 
        sdf_path, 
        target_data_path=None, 
        feature_mask=[1, 1, 1, 1, 1, 1], 
        num_samples=50, 
        noise_level=0.05, 
        device='cpu'):
    
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
    feature_distances = graph_feature_distance_list(muestra_for_model, perturbed_samples_embedding, metric='euclidean')

    # Obtener E
    E_list = [data_z.x.to(device) for data_z in perturbed_samples_embedding]

    # Lo mismo con los edges
    A_list = []
    for data_z in perturbed_samples_embedding:
    # for data_z in perturbed_samples:
        if data_z.edge_attr is not None:
            A_list.append(data_z.edge_attr.to(device))

    # --------------- HACERLO CON OPTIMIZACION DE PYTORCH ----------------------
    alfa, beta, gamma, delta, loss = obtener_argmin(feature_distances, predicciones_perturbadas, E_list, A_list, 0.01)
    # Verificar que aprendimos algo distinto de cero
    print(f"Max Alfa: {alfa.max().item():.4f}, Min Alfa: {alfa.min().item():.4f}")
    print(f"Max Beta: {beta.max().item():.4f}, Min Beta: {beta.min().item():.4f}")
    
    # ==========================================================================
    # PROCESAMIENTO DE MATRICES (Nueva Lógica)
    # ==========================================================================
    
    # 1. ALFA (Node Features) -> Filtrar -> Ordenar -> Normalizar
    node_feature_names = get_feature_names_embedding()
    alfa_sorted, row_labels_alfa = procesar_features_ordenadas(
        alfa, node_feature_names, muestra_for_model.x
    )
    col_labels_alfa = [""]

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
    col_labels_gamma = [""]

    # --- BETA (Nodos) ---
    beta_np = tensor_to_abs_numpy(beta)
    # CAMBIO: Usar normalizar_max para ser consistente con los heatmaps
    beta_np = normalizar_max(beta_np)  

    # --- DELTA (Aristas) ---
    if delta is not None:
        delta_np = tensor_to_abs_numpy(delta)
        # CAMBIO: Usar normalizar_max para evitar que una arista desaparezca si hay pocas
        delta_normalized = normalizar_max(delta_np) 
    else:
        delta_normalized = np.array([])
    
    # Preparamos el nombre del modelo para la carpeta
    model_folder_name = checkpoint_path.split('/')[-1].split('.')[0]

    # LLAMADA A LA FUNCIÓN MAESTRA
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
        algo_name="URVExplainer",
        model_name=model_folder_name  # <--- Pasamos esto para que cree la carpeta
    )

    return plotfilename

def obtener_argmin(feature_distances, predicciones_perturbadas, 
                   E_list, A_list, 
                   lr=0.05, 
                   epochs=2000, 
                   verbose=True):
    
    device = predicciones_perturbadas.device
    
    # --- DIAGNÓSTICO DE VARIANZA ---
    # Si esto es 0 o muy bajo, LIME no puede aprender nada porque el modelo
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
    # mu = nn.Parameter(torch.tensor([mean_pred], device=device, dtype=torch.float))
    mu = nn.Parameter(torch.tensor([0.0], device=device, dtype=torch.float))

    # params = [alfa, beta, mu]
    params = [alfa, beta]
    
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
    
    # Media de las distancias
    dist_mean = dists.mean()
    sigma = dist_mean if dist_mean > 0 else 1.0
    # sigma = media de las distancias
    # Calculamos weights = e^Distancia(x,z)/sigma
    # weights = e^-(distancias²) / 2 * sigma²
    weights = torch.exp(-(dists**2) / (2 * sigma**2))
    
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
        
        pred_approx = (term_nodes * scale_nodes) # + mu
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

def get_feature_names_embedding():
    return [
        "Atom Symbol", 
        "Hybridization", 
        "Degree", 
        "Total Hs", 
        "Is Aromatic", 
        "Formal Charge", 
        "Gasteiger Charge", 
        "Is Donor", 
        "Is Acceptor"
    ]

def tensor_to_abs_numpy(tensor):
    """Convierte tensor a numpy, toma valor absoluto."""
    if tensor is None: return None
    return np.abs(tensor.detach().cpu().numpy().reshape(-1, 1))

def normalizar_max(arr):
    """
    Normaliza dividiendo por el máximo absoluto.
    - El máximo será 1.0
    - El 0 real se queda en 0.
    - Mantiene la proporción real entre features.
    """
    if arr is None or len(arr) == 0: return arr
    
    # Usamos max() del valor absoluto, que ya viene calculado en 'arr'
    val_max = arr.max()
    
    if val_max == 0:
        return np.zeros_like(arr)
        
    return arr / val_max

def procesar_features_ordenadas(importance_tensor, feature_names, input_data=None):
    """
    Procesa features para Heatmaps usando Max Scaling.
    MODIFICADO: Ya no filtra features que valen 0, porque en modo Embedding 
    el 0 es una categoría válida (ej. Single Bond o Carbono).
    """
    if importance_tensor is None:
        return None, []
    
    # 1. Obtener magnitudes crudas (Valor Absoluto)
    raw_imp = tensor_to_abs_numpy(importance_tensor)
    
    # 2. SIN FILTRADO (Corrección para Embedding)
    # En embedding, siempre queremos ver todas las features (9 nodos, 2 aristas).
    # Asumimos que todas existen.
    
    # Si quieres, puedes mantener un filtrado de seguridad por dimensión,
    # pero NO por contenido igual a cero.
    filtered_imp = raw_imp
    
    # Aseguramos que feature_names sea numpy array para indexado cómodo si hiciera falta
    filtered_names = np.array(feature_names)
    
    # Safety check de dimensiones
    if len(filtered_names) != len(filtered_imp):
        logger.warning(f"Dimension mismatch in processing: Names {len(filtered_names)} vs Imp {len(filtered_imp)}")
        # Cortamos al mínimo común para evitar crash
        min_len = min(len(filtered_names), len(filtered_imp))
        filtered_names = filtered_names[:min_len]
        filtered_imp = filtered_imp[:min_len]

    if len(filtered_imp) == 0:
        return np.array([]), []

    # 3. Ordenar (Mayor a menor)
    sort_idx = np.argsort(filtered_imp.flatten())[::-1]
    
    sorted_imp = filtered_imp[sort_idx]
    sorted_names = filtered_names[sort_idx]
    
    # 4. NORMALIZAR CON MAX
    final_imp = normalizar_max(sorted_imp)
    
    return final_imp, sorted_names.tolist()

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
    
    # 3. Normalizar (Broadcasting de PyTorch hace la magia)
    #    [S, N, F] - [F] funciona directo
    normalized_stacked = (stacked - val_min) / val_range
    
    # Factor de escala para la predicción (1 / total_elementos)
    # N_elementos = stacked.shape[1]
    # scale = 1.0 / N_elementos
    
    return normalized_stacked