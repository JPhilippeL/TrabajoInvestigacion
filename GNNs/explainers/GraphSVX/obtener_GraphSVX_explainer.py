import torch
import os
import numpy as np
import logging
from rdkit import Chem

# Tus módulos existentes
from GNNs.model_tester import cargar_modelo, predecir_molecula
from GNNs.data_processing import mol_to_graph_data
from GNNs.explainers.explanation_helper import ( 
    obtener_info_real, guardar_pesos, pipeline_visualizacion_torchexplainers
)
from GNNs.explainers.GraphSVX.GraphSVX_explainer import GraphSVX

logger = logging.getLogger(__name__)
def obtener_GraphSVX_Explainer(checkpoint_path, sdf_path, target_data_path=None, batch_mode=False):
    
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
    # GraphSVX suele requerir que el grafo esté en un formato tipo batch para indexarlo,
    # aunque sea un solo grafo. Asegúrate de que las dimensiones cuadren.

    # --- 2. EJECUCIÓN GRAPHSVX ---
    logger.info(f"Ejecutando GraphSVX para {mol_name}...")
    
    # Asumo que tu clase GraphSVX se inicializa pasándole el modelo y los datos.
    # Modifica los parámetros de __init__ si tu implementación lo exige.
    explainer = GraphSVX(data=data, model=model, gpu=(device.type == 'cuda'))
    
    # Llamamos a tu método adaptado a regresión
    # Ajusta num_samples según la precisión/tiempo que necesites
    phi_list = explainer.explain_graphs(graph_indices=[0], num_samples=50, info=False)
    
    # --- 3. EXTRACCIÓN Y GUARDADO ---
    # GraphSVX no devuelve un objeto Explanation de PyG, devuelve una lista de tensores phi.
    # --- 3. EXTRACCIÓN Y GUARDADO ---
    # Convertimos el array de NumPy a un Tensor de PyTorch directamente
    phi = torch.as_tensor(phi_list[0])
    
    # El explainer necesita saber cuántas features había para cortar el tensor
    F = data.x.shape[1] 

    # MAPEO DE PESOS:
    alfa_raw  = phi[:F].detach().cpu()   # Ahora esto funcionará perfectamente
    beta_raw  = phi[F:].detach().cpu()   
    gamma_raw = None                     
    delta_raw = None
    
    model_folder_name = checkpoint_path.split('/')[-1].split('.')[0]

    # Lógica Batch
    if batch_mode:
        return {
            'mol_name': mol_name,
            'alfa': alfa_raw,
            'beta': beta_raw,
            'gamma': gamma_raw,
            'delta': delta_raw
        }
    guardar_pesos(
        alfa=alfa_raw, beta=beta_raw, gamma=gamma_raw, delta=delta_raw,
        model_name=model_folder_name, mol_name=mol_name,
        algo_name="GraphSVX"
    )

    # --- 4. VISUALIZACIÓN ---
    pred_val = predecir_molecula(model, data, device)

    # Nota: Tu pipeline_visualizacion_torchexplainers espera explanation.edge_index.
    # Como no hay objeto explanation, le pasamos directamente data.edge_index
    plotfilename = pipeline_visualizacion_torchexplainers(
        alfa_raw=alfa_raw, beta_raw=beta_raw, 
        delta_raw=delta_raw, gamma_raw=gamma_raw,
        edge_index=data.edge_index, # <-- Cambio importante aquí
        sdf_path=sdf_path,
        model=model, data=data, device=device,
        mol_name=mol_name, target_name=target_name_str,
        real_val=real_val, pred_val=pred_val,
        model_name=model_folder_name,
        algo_name="GraphSVX"
    )
    
    logger.info(f"Proceso finalizado. Gráfico GraphSVX en: {plotfilename}")
    return plotfilename