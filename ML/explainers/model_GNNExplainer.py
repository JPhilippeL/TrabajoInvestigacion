from torch_geometric.explain import Explainer, GNNExplainer
from ML.model_tester import cargar_modelo, predecir_molecula
from core.sdf_converter import parse_sdf
from ML.data_processing import mol_to_graph_data
import torch
from ui.utils import RESULTADOS_DIR
from rdkit import Chem
import os
import matplotlib.pyplot as plt
import logging
import numpy as np
from ML.explainers.model_LIME_explainer import plot_graph_with_importance
import matplotlib.pyplot as plt
import numpy as np
from ML.explainers.model_LIME_explainer import plot_graph_with_importance # Asegúrate de importar esto
from core.sdf_converter import parse_sdf # Y esto

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
    
    plotfilename = os.path.join(save_dir, f"{model_name}_{mol_name}_gnnexplainer.png")

    pred_val = predecir_molecula(model, data, device)

    # Llamamos al adaptador que usa TU función
    visualizar_custom_gnn(
        explanation=explanation, 
        sdf_path=sdf_path, 
        save_path=plotfilename,
        pred_val=pred_val,       # <--- El valor real (-3.5360)
        target_name=target_name, # <--- El nombre (targets_train)
        mol_name=mol_name,       # <--- El nombre de la molécula
        algo_name="GNNExplainer" # <--- Título
    )
    
    print(f"Explicación personalizada guardada en: {plotfilename}")
    return plotfilename

def visualizar_custom_gnn(explanation, sdf_path, save_path, pred_val, target_name, mol_name, algo_name="PGM"):
    # 1. Cargar el Grafo
    G_nx = parse_sdf(sdf_path) 
    
    # 2. Procesar Node Mask (Importancia de Átomos)
    node_mask = explanation.node_mask.detach().cpu().numpy()
    if node_mask.ndim > 1:
        node_importance = node_mask.mean(axis=1) 
    else:
        node_importance = node_mask
    
    # Normalizamos (0 a 1)
    if node_importance.max() > 0: node_importance /= node_importance.max()

    # 3. Procesar Edge Mask (Con protección para PGM)
    # Usamos .get() por seguridad
    edge_mask_tensor = explanation.get('edge_mask') 
    edge_index = explanation.edge_index.detach().cpu().numpy()

    if edge_mask_tensor is not None:
        # Caso GNNExplainer / Captum / Integrated Gradients
        edge_mask = edge_mask_tensor.detach().cpu().numpy()
        if edge_mask.max() > 0: edge_mask /= edge_mask.max()
    else:
        # Caso PGMExplainer (sin edge_mask) -> Todo a cero (gris)
        num_edges = edge_index.shape[1]
        edge_mask = np.zeros(num_edges) 

    # 4. Mapa de índices para NetworkX
    node_idx_map = {str(i): i for i in range(len(G_nx.nodes))}
    
    # 5. CONFIGURACIÓN DEL LAYOUT (SOLO GRAFO)
    # Usamos una figura cuadrada simple, sin subplots complejos
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Título dinámico
    main_title = f"{algo_name} Explanation for: {mol_name}\nModel Prediction: {pred_val:.4f} ({target_name})"
    ax.set_title(main_title, fontsize=16, fontweight='bold', pad=20)

    # 6. DIBUJAR LA MOLÉCULA
    plot_graph_with_importance(
        graph=G_nx,
        node_importance=node_importance,
        edge_importance=edge_mask,
        edge_index=edge_index,
        ax=ax,
        cmap="plasma", # O 'viridis', 'Reds', etc.
        node_idx_map=node_idx_map
    )
    
    # Forzar aspecto cuadrado para no deformar los anillos
    ax.set_aspect('equal') 

    # Guardar
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)