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