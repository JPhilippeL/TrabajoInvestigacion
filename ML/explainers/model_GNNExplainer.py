# model_GNN_explainer.py
from torch_geometric.explain import Explainer, GNNExplainer
from ML.model_tester import cargar_modelo, predecir_molecula
from ML.data_processing import mol_to_graph_data
import torch
from rdkit import Chem
import os
import numpy as np
import logging
from core.sdf_converter import parse_sdf
from ML.explainers.explanation_helper import ( 
    obtener_info_real, guardar_dashboard_explicacion,
    guardar_pesos,
    normalizar_max, get_feature_names_embedding, 
    procesar_features_ordenadas )
from ML.explainers.explanation_fidelity import calcular_curvas_fidelity, guardar_plot_fidelity

logger = logging.getLogger(__name__)

# ====================================================================
# GNN EXPLAINER (FUNCIÓN PRINCIPAL)
# ====================================================================

def obtener_GNN_Explainer(checkpoint_path, sdf_path, target_data_path=None):
    
    # 1. Cargar Modelo
    model, device, model_target_name = cargar_modelo(checkpoint_path)
    model.eval() # Importante
    
    # 2. Cargar Molécula
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None: raise ValueError(f"Error leyendo: {sdf_path}")
    
    # 3. Obtener Información Real (Target y Valor del txt)
    mol_id = os.path.basename(sdf_path).split('.')[0]
    target_name_str, real_val = obtener_info_real(target_data_path, mol_id)
    
    # Si el txt no tiene nombre del target, usamos el del modelo
    if target_name_str == "Unknown Target" and model_target_name != "Unknown":
        target_name_str = model_target_name

    # 4. Preparar Datos (Embedding Mode - 9 columnas)
    data = mol_to_graph_data(mol, mode='embedding')
    data = data.to(device)
    batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

    # Definimos los nombres de las features
    feature_names = get_feature_names_embedding()
    
    # 5. Configuración del Explainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
    )

    # 6. Ejecutar explicación
    explanation = explainer(
        x=data.x, 
        edge_index=data.edge_index, 
        edge_attr=data.edge_attr, 
        batch=batch,
        target=None
    )

    # 7. Preparar Nombres para visualización
    model_folder_name = checkpoint_path.split('/')[-1].split('.')[0]
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else mol_id
    
    # 8. Predicción
    pred_val = predecir_molecula(model, data, device)

    # 9. VISUALIZAR
    # Llamamos a la función custom que a su vez llamará a la maestra
    plotfilename = visualizar_custom_gnn(
        explanation=explanation, 
        sdf_path=sdf_path, 
        pred_val=pred_val,       
        target_name=target_name_str, 
        real_val=real_val,           
        mol_name=mol_name,       
        algo_name="GNNExplainer",
        feature_names=feature_names,       
        original_x=data.x,
        model_name=model_folder_name,
        # --- NUEVOS ARGUMENTOS PARA fidelity ---
        model=model,
        data=data,
        device=device
    )
    
    logger.info(f"Explicación GNNExplainer guardada en: {plotfilename}")
    return plotfilename

# ====================================================================
# VISUALIZACIÓN
# ====================================================================

def visualizar_custom_gnn(explanation, sdf_path, pred_val, target_name, mol_name, 
                          model_name, # <--- Necesario para guardar en la carpeta del modelo
                          real_val=None, algo_name="GNNExplainer", 
                          feature_names=None, original_x=None,
                          # Argumentos opcionales para fidelity
                          model=None, data=None, device=None):
    
    graph = parse_sdf(sdf_path) 
    
    # Manejo de máscaras
    node_mask = explanation.node_mask.detach().cpu().numpy()
    
    edge_mask_tensor = explanation.get('edge_mask')
    if edge_mask_tensor is not None:
        edge_mask = edge_mask_tensor.detach().cpu().numpy()
    else:
        edge_mask = np.zeros(explanation.edge_index.shape[1])
        
    # ==========================================================================
    # PROCESAMIENTO DE MATRICES
    # ==========================================================================

    # --- 1. ALFA (Node Features) ---
    if node_mask.ndim > 1:
        alfa_raw_tensor = torch.tensor(node_mask.mean(axis=0))
    else:
        alfa_raw_tensor = torch.tensor(node_mask)

    if original_x is not None and feature_names is not None:
        alfa_sorted, row_labels_alfa = procesar_features_ordenadas(
            alfa_raw_tensor, 
            feature_names,   
            original_x       
        )
    else:
        alfa_sorted, row_labels_alfa = None, []

    # --- 2. GAMMA (Edge Features) ---
    gamma_sorted = None
    row_labels_gamma = None

    # --- 3. BETA (Importancia Nodos - Estructura) ---
    if node_mask.ndim > 1:
        beta_np = node_mask.mean(axis=1)
    else:
        beta_np = node_mask
    
    beta_np = normalizar_max(beta_np)

    # --- 4. DELTA (Importancia Aristas) ---
    delta_normalized = normalizar_max(edge_mask)

    # ==========================================================================
    # === CÁLCULO DE fidelity (NUEVO) ===
    # ==========================================================================
    if model is not None and data is not None and device is not None:
        try:
            # 1. Calcular Curvas de FIABILIDAD
            # beta_np ya contiene la importancia de nodos calculada por GNNExplainer
            k_vals, fiab_minus = calcular_curvas_fidelity(
                model, 
                data, 
                beta_np, 
                device
            )

            # 3. Guardar (Solo pasamos datos puros)
            fiab_path = guardar_plot_fidelity(
                k_values=k_vals,
                fiab_minus=fiab_minus, 
                model_name=model_name,
                mol_name=mol_name,
                algo_name="GNNExplainer"
            )
            
            logger.info(f"Gráfico fidelity guardado en: {fiab_path}")
            
        except Exception as e:
            logger.error(f"Error calculando fidelity para GNNExplainer: {e}")
            import traceback
            traceback.print_exc()

    # ==========================================================================
    # LLAMADA A LA VISUALIZACION
    # ==========================================================================
    
    save_path = guardar_dashboard_explicacion(
        graph_obj=graph,
        edge_index=explanation.edge_index,
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
    
    return save_path