from torch_geometric.explain import Explainer, GNNExplainer
from ML.model_tester import cargar_modelo, predecir_molecula
from ML.data_processing import mol_to_graph_data, onehot_to_indices
import torch
from rdkit import Chem
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import logging
from core.sdf_converter import parse_sdf

# --- IMPORTS SOLICITADOS ---
from ui.utils import RESULTADOS_DIR, periodic_elements, hybridization_types, N_BOND_TYPES

# Importamos TODAS las funciones de LIME necesarias
from ML.explainers.model_LIME_explainer import (
    plot_graph_with_importance, 
    heatmap, 
    annotate_heatmap,
    normalizar_max,
    get_feature_names,
    procesar_features_ordenadas # <--- AÑADIDO
)

logger = logging.getLogger(__name__)

# ====================================================================
# GNN EXPLAINER (FUNCIÓN PRINCIPAL)
# ====================================================================

def obtener_GNN_Explainer(checkpoint_path, sdf_path):
    
    model, device, target_name = cargar_modelo(checkpoint_path)
    
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None: raise ValueError(f"Error leyendo: {sdf_path}")
    
    # 1. Obtenemos data en modo embedding (9 columnas)
    data = mol_to_graph_data(mol, mode='embedding')
    data = data.to(device)
    batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

    # 2. DEFINIMOS LOS NOMBRES EXACTOS (En el mismo orden que tu 'return')
    feature_names = [
        "Symbol (Idx)", 
        "Hybridization (Idx)", 
        "Degree", 
        "Total Hs", 
        "Is Aromatic", 
        "Formal Charge", 
        "Gasteiger Charge", 
        "Is Donor", 
        "Is Acceptor"
    ]
    
    # Configuración del Explainer
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

    explanation = explainer(
        x=data.x, 
        edge_index=data.edge_index, 
        edge_attr=data.edge_attr, 
        batch=batch,
        target=None
    )

    model_name = checkpoint_path.split('/')[-1].split('.')[0]
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else os.path.basename(sdf_path).split('.')[0]
    
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    save_dir = os.path.join(RESULTADOS_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    plotfilename = os.path.join(save_dir, f"{model_name}_{mol_name}_gnnexplainer.png")
    pred_val = predecir_molecula(model, data, device)

    # === DEBUG: VERIFICAR ALINEACIÓN DE COLUMNAS ===
    print("\n--- DEBUG: Verificando alineación de Features (Átomo 0) ---")
    x_sample = data.x[0].cpu().numpy() # Primer nodo
    for i, name in enumerate(feature_names):
        val = x_sample[i]
        print(f"Feature {i} [{name}]: {val}")
        
    # Guía rápida para que interpretes el print:
    # Si 'Symbol (Idx)' vale 0.15 -> ¡ERROR! El orden está mal.
    # Si 'Symbol (Idx)' vale 6.0, 7.0 u 8.0 -> ¡CORRECTO! Es un número atómico.
    # Si 'Gasteiger' vale 6.0 -> ¡ERROR! Estás leyendo el átomo como carga.
    print("-----------------------------------------------------------\n")
    # =================================================

    # 3. VISUALIZAR
    # Pasamos data.x directamente como original_x
    visualizar_custom_gnn(
        explanation=explanation, 
        sdf_path=sdf_path, 
        save_path=plotfilename,
        pred_val=pred_val,       
        target_name=target_name, 
        mol_name=mol_name,       
        algo_name="GNNExplainer",
        feature_names=feature_names,       # Lista de 9
        original_x=data.x                  # Tensor de [N, 9]
    )
    
    logger.info(f"Explicación GNNExplainer guardada en: {plotfilename}")
    return plotfilename

# ====================================================================
# VISUALIZACIÓN (Usando procesar_features_ordenadas)
# ====================================================================

def visualizar_custom_gnn(explanation, sdf_path, save_path, pred_val, target_name, mol_name, algo_name="GNNExplainer", feature_names=None, original_x=None):
    
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

    # --- 1. ALFA (Importancia de Features) ---
    # Promediamos sobre los nodos para obtener importancia global de la feature
    if node_mask.ndim > 1:
        alfa_raw_tensor = torch.tensor(node_mask.mean(axis=0))
    else:
        alfa_raw_tensor = torch.tensor(node_mask)

    # AQUÍ ESTÁ LA LÓGICA QUE PEDÍAS
    # Como alfa_raw_tensor es de tamaño 9, feature_names es 9 y original_x es [N, 9]
    # Todo encaja perfecto.
    if original_x is not None and feature_names is not None:
        alfa_sorted, row_labels_alfa = procesar_features_ordenadas(
            alfa_raw_tensor, 
            feature_names,   
            original_x       
        )
    else:
        alfa_sorted, row_labels_alfa = None, []
        
    col_labels_alfa = [""]

    # --- 2. BETA (Importancia de Nodos - Estructura) ---
    if node_mask.ndim > 1:
        beta_np = node_mask.mean(axis=1) # Promedio de las 9 features por nodo
    else:
        beta_np = node_mask
    
    beta_np = normalizar_max(beta_np)

    # --- 3. DELTA (Importancia Aristas) ---
    delta_normalized = normalizar_max(edge_mask)

    # ==========================================================================
    # PLOTTING (Igual que siempre)
    # ==========================================================================
    HEIGHT_GRAPH = 10.0   
    HEIGHT_PER_ROW = 0.4
    
    num_rows_alfa = alfa_sorted.shape[0] if alfa_sorted is not None else 0
    height_heatmaps = (num_rows_alfa * HEIGHT_PER_ROW) + 2.0 
    if height_heatmaps < 4.0: height_heatmaps = 4.0 # Altura mínima estética

    total_height = HEIGHT_GRAPH + height_heatmaps

    fig = plt.figure(figsize=(12, total_height))
    
    main_title = f"{algo_name} Explanation for: **{mol_name}**\nModel Prediction: **{pred_val:.4f}** ({target_name})"
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[HEIGHT_GRAPH, height_heatmaps], hspace=0.3)

    # Grafo
    ax_graph = fig.add_subplot(gs[0])
    node_idx_map = {str(i): i for i in range(len(graph.nodes))}
    
    plot_graph_with_importance(
        graph, 
        node_importance=beta_np.flatten(), 
        edge_importance=delta_normalized.flatten(), 
        edge_index=explanation.edge_index, 
        ax=ax_graph, 
        node_idx_map=node_idx_map,
        cmap="plasma"
    )

    # Heatmap
    ax_alfa = fig.add_subplot(gs[1])
    if alfa_sorted is not None and len(alfa_sorted) > 0:
        im_a, _ = heatmap(alfa_sorted, row_labels_alfa, col_labels_alfa, ax=ax_alfa, cmap="plasma", aspect='auto')
        annotate_heatmap(im_a, alfa_sorted, textcolors=("white", "black"))
        ax_alfa.set_title("Feature Importance (GNNExplainer)", fontsize=14)
    else:
        ax_alfa.text(0.5, 0.5, "No significant features found", ha='center')
        ax_alfa.axis('off')

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)