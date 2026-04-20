# model_GNN_explainer.py
import torch
import os
import numpy as np
import logging
from rdkit import Chem

from torch_geometric.explain import Explainer, DummyExplainer

# Tus módulos existentes
from GNNs.model_tester import cargar_modelo, predecir_molecula
from GNNs.data_processing import mol_to_graph_data
from GNNs.explainers.explanation_helper import ( 
    obtener_info_real, guardar_pesos,
    extraer_pesos_torchexplainers, pipeline_visualizacion_torchexplainers
)

logger = logging.getLogger(__name__)

def obtener_Dummy_Explainer(checkpoint_path, sdf_path, target_data_path=None, imagen=True):
    
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
    
    guardar_pesos(
        alfa=alfa_raw, beta=beta_raw, gamma=gamma_raw, delta=delta_raw,
        model_name=model_folder_name, mol_name=mol_name,
        algo_name="DummyExplainer"
    )

    if not imagen:
        return 1

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