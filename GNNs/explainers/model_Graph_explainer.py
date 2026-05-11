# model_explainer.py
from GNNs.model_tester import cargar_modelo, predecir_molecula
import torch
import torch.nn as nn
import numpy as np
from ui.utils.constants import periodic_elements, hybridization_types
import os
import sys
import logging
from GNNs.data_processing import onehot_to_indices, indices_to_onehot
from rdkit import Chem
from graph_managment.sdf_converter import parse_sdf
from GNNs.explainers.explanation_helper import ( 
    obtener_info_real, guardar_dashboard_explicacion,
    guardar_pesos, tensor_to_abs_numpy, 
    normalizar_por_norma, get_feature_names_embedding, 
    procesar_features_ordenadas )
from GNNs.explainers.graph_explainer_onehot import generate_perturbed_samples, graph_feature_distance_list

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
    
    return 0

    # guardar_pesos(alfa, beta, gamma, delta, model_folder_name,
    #               mol_name, ALGO_NAME)
    
    # # ==========================================================================
    # # PROCESAMIENTO DE MATRICES
    # # ==========================================================================
    
    # # 1. ALFA (Node Features) -> Filtrar -> Ordenar -> Normalizar
    # node_feature_names = get_feature_names_embedding()
    # alfa_sorted, row_labels_alfa = procesar_features_ordenadas(
    #     alfa, node_feature_names, muestra_for_model.x
    # )

    # # 2. GAMMA (Edge Features) -> Filtrar -> Ordenar -> Normalizar
    # # Reemplaza a Beta en el segundo heatmap
    # if muestra.edge_attr is not None:
    #     edge_feature_names = ["Bond Type", "Distance"]
        
    #     gamma_sorted, row_labels_gamma = procesar_features_ordenadas(
    #         gamma, edge_feature_names, muestra_for_model.edge_attr
    #     )
    # else:
    #     gamma_sorted = np.array([])
    #     row_labels_gamma = []

    # # --- BETA (Nodos) ---
    # beta_np = tensor_to_abs_numpy(beta)
    # # CAMBIO: Usar normalizar_por_norma para ser consistente con los heatmaps
    # beta_np = normalizar_por_norma(beta_np)  

    # # --- DELTA (Aristas) ---
    # if delta is not None:
    #     delta_np = tensor_to_abs_numpy(delta)
    #     # CAMBIO: Usar normalizar_por_norma para evitar que una arista desaparezca si hay pocas
    #     delta_normalized = normalizar_por_norma(delta_np) 
    # else:
    #     delta_normalized = np.array([])


    # # LLAMADA A LA FUNCIÓN Visualizacion
    # plotfilename = guardar_dashboard_explicacion(
    #     graph_obj=parse_sdf(sdf_path),
    #     edge_index=muestra_for_model.edge_index,
    #     node_importance=beta_np.flatten(),
    #     edge_importance=delta_normalized.flatten(),
        
    #     alfa_sorted=alfa_sorted,
    #     row_labels_alfa=row_labels_alfa,
    #     gamma_sorted=gamma_sorted,
    #     row_labels_gamma=row_labels_gamma,
        
    #     mol_name=mol_name,
    #     target_name=target_name_str,
    #     real_val=real_val,
    #     pred_val=prediccion_original,
    #     algo_name=ALGO_NAME,
    #     model_name=model_folder_name
    # )

    # return plotfilename

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