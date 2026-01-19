# model_GNN_explainer.py
import torch
import os
import numpy as np
import logging
from rdkit import Chem

from torch_geometric.explain import Explainer, GNNExplainer
from graph_managment.sdf_converter import parse_sdf

# Tus módulos existentes
from ML.model_tester import cargar_modelo, predecir_molecula
from ML.data_processing import mol_to_graph_data
from ML.explainers.explanation_helper import ( 
    obtener_info_real, guardar_dashboard_explicacion,
    guardar_pesos,
    tensor_to_abs_numpy, normalizar_max, 
    get_feature_names_embedding, procesar_features_ordenadas 
)
from ML.explainers.explanation_fidelity import calcular_curvas_fidelity_general, guardar_plot_fidelity

logger = logging.getLogger(__name__)

# ====================================================================
# 3. FUNCIÓN PRINCIPAL (ENTRY POINT)
# ====================================================================
def obtener_GNN_Explainer(checkpoint_path, sdf_path, target_data_path=None, imagen = True):
    
    # --- 1. CARGA DE RECURSOS ---
    model, device, model_target_name = cargar_modelo(checkpoint_path)
    model.eval()
    
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    mol_id = os.path.basename(sdf_path).split('.')[0]
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else mol_id
    
    # Info Target
    target_name_str, real_val = obtener_info_real(target_data_path, mol_id)
    if target_name_str == "Unknown Target" and model_target_name != "Unknown":
        target_name_str = model_target_name

    # Grafo
    data = mol_to_graph_data(mol, mode='embedding').to(device)
    batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

    # --- 2. EJECUCIÓN GNNEXPLAINER ---
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(mode='regression', task_level='graph', return_type='raw'),
    )

    logger.info(f"Ejecutando GNNExplainer para {mol_name}...")
    explanation = explainer(
        x=data.x, edge_index=data.edge_index, 
        edge_attr=data.edge_attr, batch=batch, target=None
    )

    # --- 3. EXTRACCIÓN Y GUARDADO DE PESOS ---
    alfa_raw, beta_raw, gamma_raw, delta_raw = extraer_pesos_gnn_explainer(explanation)
    
    model_folder_name = checkpoint_path.split('/')[-1].split('.')[0]
    
    # Guardamos los tensores crudos (con el orden correcto)
    guardar_pesos(
        alfa=alfa_raw, beta=beta_raw, gamma=gamma_raw, delta=delta_raw,
        model_name=model_folder_name, mol_name=mol_name,
        algo_name="GNNExplainer"
    )

    if imagen == False:
        logger.info("Pesos guardados, no se hizo imagen")
        return 1

    # --- 4. ANÁLISIS Y VISUALIZACIÓN ---
    pred_val = predecir_molecula(model, data, device)

    plotfilename = ejecutar_pipeline_visualizacion(
        alfa_raw=alfa_raw, beta_raw=beta_raw, 
        delta_raw=delta_raw, gamma_raw=gamma_raw,
        edge_index=explanation.edge_index,
        sdf_path=sdf_path,
        model=model, data=data, device=device,
        mol_name=mol_name, target_name=target_name_str,
        real_val=real_val, pred_val=pred_val,
        model_name=model_folder_name
    )
    
    logger.info(f"Proceso finalizado. Gráfico en: {plotfilename}")
    return plotfilename

# ====================================================================
# 1. LOGICA DE EXTRACCIÓN (HELPER)
# ====================================================================
def extraer_pesos_gnn_explainer(explanation):
    """
    Convierte la salida compleja de GNNExplainer en los 4 tensores base (Raw).
    Mantiene el orden original de los nodos (índices).
    """
    node_mask = explanation.node_mask
    edge_mask = explanation.edge_mask

    # --- ALFA (Features Importancia Global) ---
    # Promedio por columna (features). Mantiene dimensión [Num_Features]
    if node_mask.ndim > 1:
        alfa_raw = node_mask.mean(dim=0) 
    else:
        alfa_raw = node_mask

    # --- BETA (Nodos Importancia Estructural) ---
    # Promedio por fila (nodos). Mantiene dimensión [Num_Nodos]
    # CRÍTICO: Mantiene el orden de índices para Fidelity
    if node_mask.ndim > 1:
        beta_raw = node_mask.mean(dim=1)
    else:
        beta_raw = node_mask

    # --- DELTA (Aristas) ---
    delta_raw = edge_mask if edge_mask is not None else None
    
    # --- GAMMA (Features Arista) ---
    # GNNExplainer estándar no suele devolver feature mask para aristas
    gamma_raw = None 

    return alfa_raw, beta_raw, gamma_raw, delta_raw


# ====================================================================
# 2. PIPELINE DE ANÁLISIS Y VISUALIZACIÓN
# ====================================================================
def ejecutar_pipeline_visualizacion(
    alfa_raw, beta_raw, delta_raw, gamma_raw,
    edge_index, sdf_path, model, data, device,
    mol_name, target_name, real_val, pred_val, 
    model_name, algo_name="GNNExplainer"
):
    """
    Toma los pesos CRUDOS y coordina:
    1. Cálculo de Fidelity (con orden original).
    2. Procesamiento para Heatmaps (ordenar y etiquetar).
    3. Generación del Dashboard.
    """
    
    # A. PREPARACIÓN DE DATOS BASE
    graph_obj = parse_sdf(sdf_path)
    
    # Convertimos a numpy normalizado (MAX) para pintar y para el threshold de fidelity
    beta_np = normalizar_max(tensor_to_abs_numpy(beta_raw))
    delta_normalized = normalizar_max(tensor_to_abs_numpy(delta_raw)) if delta_raw is not None else np.array([])

    # ---------------------------------------------------------
    # B. CÁLCULO DE FIDELITY (Sobre datos alineados con grafos)
    # ---------------------------------------------------------
    if model is not None and data is not None:
        try:
            logger.info("Calculando curvas de fidelity...")
            k_vals, fiab_minus = calcular_curvas_fidelity_general(
                model, data, beta_np, device
            )
            guardar_plot_fidelity(
                k_values=k_vals, fiab_minus=fiab_minus, 
                model_name=model_name, mol_name=mol_name, algo_name=algo_name
            )
        except Exception as e:
            logger.error(f"Error calculando fidelity: {e}")

    # ---------------------------------------------------------
    # C. PROCESAMIENTO PARA HEATMAPS (Aquí sí alteramos el orden)
    # ---------------------------------------------------------
    # Solo necesario para ALFA y GAMMA que llevan etiquetas de texto
    feature_names = get_feature_names_embedding()
    
    # ALFA
    if alfa_raw is not None:
        alfa_sorted, row_labels_alfa = procesar_features_ordenadas(
            alfa_raw, feature_names, data.x
        )
    else:
        alfa_sorted, row_labels_alfa = None, []

    # GAMMA
    if gamma_raw is not None:
        gamma_sorted, row_labels_gamma = procesar_features_ordenadas(
            gamma_raw, ["Bond Type", "Dist"], data.edge_attr
        )
    else:
        gamma_sorted, row_labels_gamma = None, []

    # ---------------------------------------------------------
    # D. GENERAR DASHBOARD FINAL
    # ---------------------------------------------------------
    plot_path = guardar_dashboard_explicacion(
        graph_obj=graph_obj,
        edge_index=edge_index,
        node_importance=beta_np.flatten(),
        edge_importance=delta_normalized.flatten(),
        alfa_sorted=alfa_sorted,
        row_labels_alfa=row_labels_alfa,
        gamma_sorted=gamma_sorted,
        row_labels_gamma=row_labels_gamma,
        mol_name=mol_name,
        target_name=target_name,
        real_val=real_val,
        pred_val=pred_val,
        algo_name=algo_name,
        model_name=model_name
    )
    
    return plot_path