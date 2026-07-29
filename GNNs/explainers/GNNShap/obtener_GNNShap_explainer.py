import torch
import os
import logging
from rdkit import Chem

# Tus módulos existentes
from GNNs.model_tester import cargar_modelo, predecir_molecula
from GNNs.data_processing import mol_to_graph_data
from GNNs.explainers.explanation_helper import ( 
    obtener_info_real, guardar_pesos, pipeline_visualizacion_torchexplainers
)
# Asegúrate de importar la clase GNNShap (o como la tengas nombrada en tu proyecto)
from GNNs.explainers.GNNShap.GNNshap_explainer import GNNShapExplainer

logger = logging.getLogger(__name__)

def obtener_GNNShap_Explainer(checkpoint_path, sdf_path, target_data_path=None, batch_mode=False):
    
    # --- 1. CARGA DE RECURSOS ---
    model, device, model_target_name = cargar_modelo(checkpoint_path)
    model.eval()
    
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    mol_id = os.path.basename(sdf_path).split('.')[0]
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else mol_id
    
    target_name_str, real_val = obtener_info_real(target_data_path, mol_id)
    if target_name_str == "Unknown Target" and model_target_name != "Unknown":
        target_name_str = model_target_name

    # Generamos el grafo de la molécula
    data = mol_to_graph_data(mol, mode='embedding').to(device)

    # --- 2. EJECUCIÓN GNNSHAP ---
    logger.info(f"Ejecutando GNNShap para {mol_name}...")
    
    # Asumimos que inicializas GNNShap pasándole el modelo, los datos, y quizás num_hops.
    explainer = GNNShapExplainer(model=model, data=data, device=device, num_hops=5)
    
    # Usamos el método de regresión adaptado
    explanation = explainer.explain_regression(
        node_idx=0, 
        nsamples=200,          
        batch_size=512, 
        solver_name="WLSSolver"
    )
    
    # --- 3. EXTRACCIÓN Y GUARDADO ---
    # GNNShap nos devuelve un objeto con shap_vals correspondientes a las aristas evaluadas[cite: 1].
    shap_vals = torch.as_tensor(explanation.shap_vals, dtype=torch.float32).detach().cpu()
    
    # MAPEO DE PESOS (Corregido para fidelidad al original):
    alfa_raw  = None      # Nodos features
    beta_raw  = None      # Nodos
    gamma_raw = None      # Aristas features
    delta_raw = shap_vals # Aristas (edges)
    
    model_folder_name = checkpoint_path.split('/')[-1].split('.')[0]

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
        algo_name="GNNShap"
    )

    # --- 4. VISUALIZACIÓN ---
    pred_val = predecir_molecula(model, data, device)

    # Usamos el edge_index que devuelve el propio explicador (sub_edge_index)
    plotfilename = pipeline_visualizacion_torchexplainers(
        alfa_raw=alfa_raw, beta_raw=beta_raw, 
        delta_raw=delta_raw, gamma_raw=gamma_raw,
        edge_index=explanation.sub_edge_index.detach().cpu(), 
        sdf_path=sdf_path,
        model=model, data=data, device=device,
        mol_name=mol_name, target_name=target_name_str,
        real_val=real_val, pred_val=pred_val,
        model_name=model_folder_name,
        algo_name="GNNShap"
    )
    
    logger.info(f"Proceso finalizado. Gráfico GNNShap en: {plotfilename}")
    return plotfilename