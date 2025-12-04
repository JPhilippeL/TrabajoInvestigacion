from torch_geometric.explain import Explainer, GNNExplainer
from ML.model_tester import cargar_modelo
from core.sdf_converter import parse_sdf
from ML.data_processing import mol_to_graph_data
import torch
from ui.utils import RESULTADOS_DIR
from rdkit import Chem
import os
import matplotlib.pyplot as plt
import logging
from ML.explainers.model_LIME_explainer import plot_graph_with_importance

logger = logging.getLogger(__name__)

def obtener_GNN_Explainer(checkpoint_path, sdf_path):
    # ... (Carga de modelo y datos igual que antes) ...
    model, device, target_name = cargar_modelo(checkpoint_path)
    
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None: raise ValueError(f"Error leyendo: {sdf_path}")
    
    data = mol_to_graph_data(mol, mode='embedding')
    data = data.to(device)
    batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

    # ... (Configuración del Explainer igual que antes) ...
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

    # Ejecutar explicación
    explanation = explainer(
        x=data.x, 
        edge_index=data.edge_index, 
        edge_attr=data.edge_attr, 
        batch=batch,
        target=None
    )

    # --- CAMBIO AQUÍ: VISUALIZACIÓN PERSONALIZADA ---
    model_name = checkpoint_path.split('/')[-1].split('.')[0]
    
    # Recuperamos el nombre de la molécula para el archivo
    suppl = Chem.SDMolSupplier(sdf_path)
    mol_obj = next(m for m in suppl if m is not None)
    mol_name = mol_obj.GetProp("_Name") if mol_obj.HasProp("_Name") else os.path.basename(sdf_path).split('.')[0]
    
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    save_dir = os.path.join(RESULTADOS_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    plotfilename = os.path.join(save_dir, f"{model_name}_{mol_name}_gnnexplainer_custom.png")

    # Llamamos al adaptador que usa TU función
    visualizar_custom_gnn(explanation, sdf_path, plotfilename)
    
    print(f"Explicación personalizada guardada en: {plotfilename}")
    return plotfilename

def visualizar_custom_gnn(explanation, sdf_path, save_path):
    # 1. Cargar el Grafo NetworkX (Usando tu parse_sdf existente)
    # Este grafo tiene las posiciones (pos) y tipos de enlace (bond_type)
    G_nx = parse_sdf(sdf_path) 
    
    # 2. Procesar Node Mask (Importancia de Átomos)
    # GNNExplainer devuelve [N_nodos, N_features]. 
    # Necesitamos reducirlo a [N_nodos] (un valor por átomo).
    node_mask = explanation.node_mask.detach().cpu().numpy()
    
    # Si tienes varias features por átomo, hacemos la media o el máximo
    if node_mask.ndim > 1:
        # axis=1 colapsa las columnas de features a un solo número
        node_importance = node_mask.mean(axis=1) 
    else:
        node_importance = node_mask
        
    # Normalizamos entre 0 y 1 para que los colores salgan bien
    if node_importance.max() > 0:
        node_importance /= node_importance.max()

    # 3. Procesar Edge Mask (Importancia de Enlaces)
    edge_mask = explanation.edge_mask.detach().cpu().numpy()
    edge_index = explanation.edge_index.detach().cpu().numpy()
    
    # Normalizamos aristas
    if edge_mask.max() > 0:
        edge_mask /= edge_mask.max()

    # 4. Crear Mapa de Índices
    # Tu parse_sdf usa strings "0", "1" como IDs, pero PyTorch usa ints 0, 1.
    # Creamos un mapa para que la función sepa quien es quien.
    node_idx_map = {str(i): i for i in range(len(G_nx.nodes))}

    # 5. Llamar a TU función de ploteo
    fig, ax = plt.subplots(figsize=(10, 10))
    
    plot_graph_with_importance(
        graph=G_nx,
        node_importance=node_importance,
        edge_importance=edge_mask,
        edge_index=edge_index,
        ax=ax,
        node_idx_map=node_idx_map
    )
    
    # Guardar
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns # Opcional, pero ayuda con las barras. Si no tienes, usa plt.bar

def visualizar_custom_gnn2(explanation, sdf_path, save_path):
    # --- 1. PREPARACIÓN DE DATOS (Igual que antes) ---
    G_nx = parse_sdf(sdf_path) 
    
    # Mascaras en numpy
    node_mask = explanation.node_mask.detach().cpu().numpy()
    edge_mask = explanation.edge_mask.detach().cpu().numpy()
    edge_index = explanation.edge_index.detach().cpu().numpy()

    # Importancia Estructural (Para colorear el grafo)
    if node_mask.ndim > 1:
        node_importance_graph = node_mask.mean(axis=1) # Promedio por átomo para el color
    else:
        node_importance_graph = node_mask
    if node_importance_graph.max() > 0: node_importance_graph /= node_importance_graph.max()
    
    if edge_mask.max() > 0: edge_mask_norm = edge_mask / edge_mask.max()
    else: edge_mask_norm = edge_mask

    node_idx_map = {str(i): i for i in range(len(G_nx.nodes))}

    # --- 2. AGREGACIÓN DE IMPORTANCIA DE FEATURES (Para las barras de abajo) ---
    
    # A) NOMBRES DE LAS FEATURES (Deben coincidir con tu función mol_to_graph)
    # He copiado los de tu imagen de LIME. Verifica si el orden es correcto en tu tensor .x
    feature_names = [
        "GasteigerCharge", "IsHDonor", "IsHAcceptor", "IsAromatic", 
        "Hybrid_SP", "Hybrid_SP2", "Hybrid_SP3", "Hybrid_S", # Asumiendo one-hot de hibridación
        "Atom_H", "Atom_C", "Atom_N", "Atom_O", "Atom_F", "Atom_Cl", "Atom_S", # Asumiendo one-hot de átomos
        "Degree" # Si usas degree
    ]
    # IMPORTANTE: Si tu modelo tiene menos o más features, ajusta esta lista o corta/rellena
    num_feats_model = node_mask.shape[1]
    current_names = feature_names[:num_feats_model] if len(feature_names) >= num_feats_model else [f"Feat_{i}" for i in range(num_feats_model)]

    # Calculamos importancia promedio de cada feature en toda la molécula
    feat_importance_vals = node_mask.mean(axis=0)
    # Normalizamos
    if feat_importance_vals.max() > 0: feat_importance_vals /= feat_importance_vals.max()
    
    df_feat = pd.DataFrame({'Feature': current_names, 'Importance': feat_importance_vals})
    df_feat = df_feat.sort_values(by='Importance', ascending=False).head(10) # Top 10

    # B) IMPORTANCIA DE TIPOS DE ENLACE (Aproximación)
    # Cruzamos la máscara de aristas con el tipo de enlace del grafo
    bond_types_imp = {}
    for i in range(len(edge_mask)):
        u, v = str(edge_index[0, i]), str(edge_index[1, i])
        # Buscamos el tipo en el grafo NetworkX
        edge_data = G_nx.get_edge_data(u, v)
        if edge_data:
            b_type = str(edge_data.get('bond_type', 'Unknown'))
            weight = edge_mask[i]
            bond_types_imp[b_type] = bond_types_imp.get(b_type, 0.0) + weight

    # Normalizar
    max_bond = max(bond_types_imp.values()) if bond_types_imp else 1
    df_edge = pd.DataFrame([
        {'Type': k, 'Importance': v / max_bond} for k, v in bond_types_imp.items()
    ]).sort_values(by='Importance', ascending=False)

    # --- 3. CONFIGURACIÓN DEL LAYOUT (GRID) ---
    # Creamos una figura grande con diseño personalizado
    fig = plt.figure(figsize=(12, 12)) # Tamaño similar a tu imagen LIME
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1]) # 3 partes grafo, 1 parte barras
    
    # Ax1: El Grafo (Ocupa toda la parte superior)
    ax_graph = fig.add_subplot(gs[0, :]) 
    ax_graph.set_title("GNNExplainer Structure Explanation", fontsize=16, fontweight='bold')
    
    # Ax2: Feature Importance (Abajo Izquierda)
    ax_feat = fig.add_subplot(gs[1, 0])
    
    # Ax3: Edge Importance (Abajo Derecha)
    ax_edge = fig.add_subplot(gs[1, 1])

    # --- 4. DIBUJAR EL GRAFO ---
    plot_graph_with_importance(
        graph=G_nx,
        node_importance=node_importance_graph,
        edge_importance=edge_mask_norm,
        edge_index=edge_index,
        ax=ax_graph, # Pasamos el eje específico
        cmap="plasma", # El de tu imagen LIME parece 'plasma' o 'magma'
        node_idx_map=node_idx_map
    )
    # Forzamos aspecto solo en el grafo
    ax_graph.set_aspect('equal')

    # --- 5. DIBUJAR BARRAS DE FEATURES ---
    if not df_feat.empty:
        sns.barplot(data=df_feat, x="Importance", y="Feature", ax=ax_feat, palette="plasma")
        ax_feat.set_title("Node Feature Importance", fontsize=12)
        ax_feat.set_xlabel("")
        ax_feat.set_xlim(0, 1)
    
    # --- 6. DIBUJAR BARRAS DE ENLACES ---
    if not df_edge.empty:
        sns.barplot(data=df_edge, x="Importance", y="Type", ax=ax_edge, palette="plasma")
        ax_edge.set_title("Edge Type Importance", fontsize=12)
        ax_edge.set_xlabel("")
        ax_edge.set_xlim(0, 1)

    # Guardar
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)