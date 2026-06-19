# model_GNN_explainer.py
import torch
import os
import numpy as np
import logging
from rdkit import Chem

from torch_geometric.explain import Explainer, DummyExplainer, GNNExplainer, CaptumExplainer

# Tus módulos existentes
from GNNs.model_tester import cargar_modelo, predecir_molecula
from GNNs.data_processing import mol_to_graph_data
from GNNs.explainers.explanation_helper import ( 
    obtener_info_real, guardar_pesos,
    extraer_pesos_torchexplainers, pipeline_visualizacion_torchexplainers,
    tensor_to_abs_numpy, normalizar_por_l2, procesar_features_ordenadas
)
from ui.utils.constants import *
logger = logging.getLogger(__name__)

# ----------------------    DUMMY EXPLAINER -------------------------
def obtener_Dummy_Explainer(checkpoint_path, sdf_path, target_data_path=None, batch_mode=False):
    
    # --- 1. CARGA DE RECURSOS ---
    model, device, model_target_name = cargar_modelo(checkpoint_path)
    model.eval()
    
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    mol_id = os.path.basename(sdf_path).split('.')[0]
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else mol_id
    
    target_name_str, real_val = obtener_info_real(target_data_path, mol_id)
    if target_name_str == "Unknown Target" and model_target_name != "Unknown":
        target_name_str = model_target_name

    data = mol_to_graph_data(mol, mode='embedding').to(device)
    batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

    # --- 2. EJECUCIÓN DUMMY EXPLAINER ---
    # Es la misma clase base, solo cambia el algoritmo
    explainer = Explainer(
        model=model,
        algorithm=DummyExplainer(), # Genera ruido aleatorio
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(mode='regression', task_level='graph', return_type='raw'),
    )

    logger.info(f"Ejecutando DummyExplainer (Baseline Aleatorio) para {mol_name}...")
    
    explanation = explainer(
        x=data.x, edge_index=data.edge_index, 
        edge_attr=data.edge_attr, batch=batch, target=None
    )

    # --- 3. EXTRACCIÓN Y GUARDADO ---
    alfa_raw, beta_raw, gamma_raw, delta_raw = extraer_pesos_torchexplainers(explanation)
    model_folder_name = checkpoint_path.split('/')[-1].split('.')[0]

    # NUEVA LÓGICA: Si es batch, preparamos el diccionario y retornamos INMEDIATAMENTE
    if batch_mode:
        return {
            'mol_name': mol_name, # Para usarlo como llave en el bucle
            'alfa': alfa_raw.detach().cpu() if alfa_raw is not None else None,
            'beta': beta_raw.detach().cpu() if beta_raw is not None else None,
            'gamma': gamma_raw.detach().cpu() if gamma_raw is not None else None,
            'delta': delta_raw.detach().cpu() if delta_raw is not None else None
        }
    
    guardar_pesos(
        alfa=alfa_raw, beta=beta_raw, gamma=gamma_raw, delta=delta_raw,
        model_name=model_folder_name, mol_name=mol_name,
        algo_name="DummyExplainer"
    )

    # --- 4. VISUALIZACIÓN ---
    pred_val = predecir_molecula(model, data, device)

    plotfilename = pipeline_visualizacion_torchexplainers(
        alfa_raw=alfa_raw, beta_raw=beta_raw, 
        delta_raw=delta_raw, gamma_raw=gamma_raw,
        edge_index=explanation.edge_index,
        sdf_path=sdf_path,
        model=model, data=data, device=device,
        mol_name=mol_name, target_name=target_name_str,
        real_val=real_val, pred_val=pred_val,
        model_name=model_folder_name,
        algo_name="DummyExplainer"
    )
    
    logger.info(f"Proceso finalizado. Gráfico Dummy en: {plotfilename}")
    return plotfilename

# ----------------------    GNN EXPLAINER -------------------------
def obtener_GNN_Explainer(checkpoint_path, sdf_path, target_data_path=None, batch_mode=False):
    
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
    alfa_raw, beta_raw, gamma_raw, delta_raw = extraer_pesos_torchexplainers(explanation)
    
    model_folder_name = checkpoint_path.split('/')[-1].split('.')[0]

    # NUEVA LÓGICA: Si es batch, preparamos el diccionario y retornamos INMEDIATAMENTE
    if batch_mode:
        alfa_norm = procesar_features_ordenadas(alfa_raw, NODE_FEATURES_NAMES_EMBEDDING)
        beta_raw = tensor_to_abs_numpy(beta_raw)
        gamma_norm = procesar_features_ordenadas(gamma_raw, EDGE_FEATURE_NAMES_EMBEDDING) if gamma_raw is not None else np.array([])
        delta_raw = tensor_to_abs_numpy(delta_raw) if delta_raw is not None else np.array([])

        beta_norm = normalizar_por_l2(beta_raw)
        delta_norm = normalizar_por_l2(delta_raw)
        return {
            'mol_name': mol_name, 
            'alfa': alfa_norm,  
            'beta': beta_norm,
            'gamma': gamma_norm ,
            'delta': delta_norm
        }
    
    # Guardamos los tensores crudos (con el orden correcto)
    guardar_pesos(
        alfa=alfa_raw, beta=beta_raw, gamma=gamma_raw, delta=delta_raw,
        model_name=model_folder_name, mol_name=mol_name,
        algo_name="GNNExplainer"
    )

    # --- 4. ANÁLISIS Y VISUALIZACIÓN ---
    pred_val = predecir_molecula(model, data, device)

    plotfilename = pipeline_visualizacion_torchexplainers(
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

# ----------------------    CAPTUM EXPLAINER -------------------------
def obtener_Captum_Explainer(checkpoint_path, sdf_path, target_data_path=None, batch_mode=False, captum_method='ShapleyValueSampling'):
    """
    Ejecuta un explicador basado en la librería Captum.
    Opciones recomendadas para captum_method: 
    - 'IntegratedGradients' (Baseline sólido)
    - 'InputXGradient' (Muy rápido)
    - 'ShapleyValueSampling' (Aproximación SHAP)
    """
    
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

    # --- 2. EJECUCIÓN CAPTUM EXPLAINER ---
    # --- 2. EJECUCIÓN CAPTUM EXPLAINER ---
    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer(captum_method),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object', # <--- CAMBIADO DE 'attributes' A 'object'
        model_config=dict(mode='regression', task_level='graph', return_type='raw'),
    )

    logger.info(f"Ejecutando CaptumExplainer ({captum_method}) para {mol_name}...")
    
    # NOTA CRÍTICA PARA CAPTUM: A diferencia de GNNExplainer, los métodos basados 
    # en gradiente de Captum a veces necesitan saber el índice de la salida. 
    # Si tu modelo devuelve un tensor de [1, 1], usa target=0. 
    # Si falla, cámbialo a target=None como lo tenías.
    target_idx = 0 
    
    explanation = explainer(
        x=data.x, edge_index=data.edge_index, 
        edge_attr=data.edge_attr, batch=batch, target=target_idx
    )

    # --- 3. EXTRACCIÓN Y GUARDADO DE PESOS ---
    # Reutilizamos tu función helper exacta porque la API de PyG estandariza la salida
    alfa_raw, beta_raw, gamma_raw, delta_raw = extraer_pesos_torchexplainers(explanation)
    
    model_folder_name = checkpoint_path.split('/')[-1].split('.')[0]
    algo_name_full = f"Captum_{captum_method}"

    # NUEVA LÓGICA: Si es batch, preparamos el diccionario y retornamos INMEDIATAMENTE
    if batch_mode:
        return {
            'mol_name': mol_name, # Para usarlo como llave en el bucle
            'alfa': alfa_raw.detach().cpu() if alfa_raw is not None else None,
            'beta': beta_raw.detach().cpu() if beta_raw is not None else None,
            'gamma': gamma_raw.detach().cpu() if gamma_raw is not None else None,
            'delta': delta_raw.detach().cpu() if delta_raw is not None else None
        }
    
    # Guardamos los tensores crudos
    guardar_pesos(
        alfa=alfa_raw, beta=beta_raw, gamma=gamma_raw, delta=delta_raw,
        model_name=model_folder_name, mol_name=mol_name,
        algo_name=algo_name_full
    )

    # --- 4. ANÁLISIS Y VISUALIZACIÓN ---
    pred_val = predecir_molecula(model, data, device)

    # Reutilizamos tu pipeline de visualización tal cual
    plotfilename = pipeline_visualizacion_torchexplainers(
        alfa_raw=alfa_raw, beta_raw=beta_raw, 
        delta_raw=delta_raw, gamma_raw=gamma_raw,
        edge_index=explanation.edge_index,
        sdf_path=sdf_path,
        model=model, data=data, device=device,
        mol_name=mol_name, target_name=target_name_str,
        real_val=real_val, pred_val=pred_val,
        model_name=model_folder_name,
        algo_name=algo_name_full # Le pasamos el nombre específico para el dashboard
    )
    
    logger.info(f"Proceso finalizado. Gráfico en: {plotfilename}")
    return plotfilename


def obtener_GNN_Explainer_PT(checkpoint_path, data_indices, batch_mode=False):
    
    # --- 1. CARGA DE RECURSOS ---
    model, device, model_target_name = cargar_modelo(checkpoint_path)
    model.eval()
    
    # Extraemos la información directamente del objeto Data pasado por parámetro
    mol_name = getattr(data_indices, 'name', 'Unknown_Mol')
    real_val = getattr(data_indices, 'y', None)
    
    # Pasamos el grafo al dispositivo correspondiente
    data = data_indices.to(device)
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
    
    # Nos aseguramos de manejar el caso en el que edge_attr pueda no existir en el grafo
    edge_attr = getattr(data, 'edge_attr', None)
    
    explanation = explainer(
        x=data.x, 
        edge_index=data.edge_index, 
        edge_attr=edge_attr, 
        batch=batch, 
        target=None
    )

    # --- 3. EXTRACCIÓN Y GUARDADO DE PESOS ---
    alfa_raw, beta_raw, gamma_raw, delta_raw = extraer_pesos_torchexplainers(explanation)
    
    model_folder_name = checkpoint_path.split('/')[-1].split('.')[0]

    # LÓGICA DE BATCH MODE
    if batch_mode:
        alfa_norm = procesar_features_ordenadas(alfa_raw, NODE_FEATURES_NAMES_EMBEDDING)
        beta_raw = tensor_to_abs_numpy(beta_raw)
        gamma_norm = procesar_features_ordenadas(gamma_raw, EDGE_FEATURE_NAMES_EMBEDDING) if gamma_raw is not None else np.array([])
        delta_raw = tensor_to_abs_numpy(delta_raw) if delta_raw is not None else np.array([])

        beta_norm = normalizar_por_l2(beta_raw)
        delta_norm = normalizar_por_l2(delta_raw)
        
        return {
            'mol_name': mol_name, 
            'alfa': alfa_norm,  
            'beta': beta_norm,
            'gamma': gamma_norm ,
            'delta': delta_norm
        }