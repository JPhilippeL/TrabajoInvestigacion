import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from ui.utils import RESULTADOS_DIR

# Constante N: Número máximo de nodos a evaluar en la curva
MAX_NODES_FIDELITY = 15

def calcular_curvas_fidelity(model, data, node_importance, device):
    """
    Calcula las curvas eliminando FÍSICAMENTE los nodos menos importantes y sus aristas.
    """
    model.eval()
    data = data.to(device)
    num_nodes = data.x.shape[0]
    
    # Aseguramos que importance sea numpy y aplanado
    if torch.is_tensor(node_importance):
        imp = node_importance.detach().cpu().numpy().flatten()
    else:
        imp = np.array(node_importance).flatten()

    imp = np.abs(imp)

    # === ORDEN ASCENDENTE (De Menor a Mayor Importancia) ===
    sorted_indices = np.argsort(imp).copy() 
    # =======================================================

    # Predicción original
    with torch.no_grad():
        pred_original = model(data.x, data.edge_index, data.edge_attr, data.batch)
        val_orig = pred_original.item()

    fiab_list = []
    k_values = []

    limit = min(num_nodes, MAX_NODES_FIDELITY)

    for k in range(limit + 1):
        k_values.append(k)
        
        # Índices de los nodos (ruido) a eliminar en esta iteración
        current_k_indices = sorted_indices[:k] 

        # --- Fidelity (Physical Removal) ---
        if k == 0:
            # Si k=0, no eliminamos nada
            data_minus = data
        else:
            # Aquí llamamos a la función de eliminación real
            data_minus = eliminar_nodos_y_conexiones(data, current_k_indices)
        
        with torch.no_grad():
            # Nota: data_minus ahora tiene menos nodos y edge_index re-mapeado
            # Es posible que el modelo falle si espera un tamaño fijo, 
            # pero en GNNs estándar (GCN, GAT, GraphSAGE) esto funciona bien.
            if data_minus.x.shape[0] == 0:
                # Caso extremo: se borraron todos los nodos
                # Definir comportamiento (ej: predicción 0 o mantener la anterior)
                val_minus = 0.0 
            else:
                pred_minus = model(data_minus.x, data_minus.edge_index, data_minus.edge_attr, data_minus.batch)
                val_minus = pred_minus.item()
            
        diff_minus = abs(val_orig - val_minus)
        fiab_minus = np.exp(-diff_minus)
        fiab_list.append(fiab_minus)

    return k_values, fiab_list

def guardar_plot_fidelity(k_values, fiab_minus, model_name, mol_name, algo_name="Explainer"):
    """
    Genera el gráfico con los colores invertidos:
    - Fidelity- (Debe ser alto) -> VERDE
    - Fidelity+ (Debe ser bajo) -> ROJO
    """
    
    # 1. Sanitizar nombre
    safe_mol_name = "".join([c for c in mol_name if c.isalnum() or c in (' ', '_', '-')]).strip()
    safe_mol_name = safe_mol_name.replace(" ", "_")
    
    # 2. Nombre de archivo
    filename = f"FIDELITY_{model_name}_{safe_mol_name}_{algo_name}.png"
    
    # 3. Directorios
    base_model_dir = os.path.join(RESULTADOS_DIR, model_name)
    fidelity_dir = os.path.join(base_model_dir, "Fidelity")
    os.makedirs(fidelity_dir, exist_ok=True)
    
    full_save_path = os.path.join(fidelity_dir, filename)

    # 4. AUC
    auc_minus = np.trapezoid(fiab_minus, k_values)
    
    plt.figure(figsize=(10, 6))
    
    # Etiquetas
    label_minus = f'Fidelity (Remove ONLY Low Imp.)\nAUC: {auc_minus:.2f} (Ideal: High)'

    # === COLORES MODIFICADOS ===
    # Fidelity -> Verde (Queremos que se mantenga alto)
    plt.plot(k_values, fiab_minus, marker='x', label=label_minus, color='green', linestyle='--', linewidth=2)

    plt.title(f"Noise Robustness Analysis ({algo_name}): {mol_name}", fontsize=12, fontweight='bold')
    plt.xlabel("K (Number of Low Importance Nodes modified)", fontsize=10)
    plt.ylabel("Prediction Similarity $e^{-|Error|}$", fontsize=10)
    
    plt.ylim(-0.05, 1.05) 
    plt.axhline(1, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    # Rellenos (Match con los colores de las líneas)
    plt.fill_between(k_values, fiab_minus, color='green', alpha=0.1)

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0,
               fontsize=9, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.xticks(k_values)
    plt.tight_layout()
    
    plt.savefig(full_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return full_save_path

def eliminar_nodos_y_conexiones(data, indices_a_eliminar):
    """
    Crea un nuevo objeto Data eliminando los nodos especificados y
    todas las aristas conectadas a ellos, re-indexando el grafo.
    """
    num_nodes = data.x.shape[0]
    device = data.x.device
    
    # 1. Crear máscara booleana de los nodos que se quedan (KEEP)
    subset_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    subset_mask[indices_a_eliminar] = False
    
    # 2. Filtrar aristas y re-etiquetar nodos (relabel_nodes=True es la clave)
    # Esto asegura que si borras el nodo 0, el nodo 1 pasa a ser el nuevo 0 en edge_index
    edge_index, edge_attr = subgraph(
        subset_mask, 
        data.edge_index, 
        data.edge_attr, 
        relabel_nodes=True, 
        num_nodes=num_nodes
    )
    
    # 3. Filtrar características de los nodos (x) y batch
    x = data.x[subset_mask]
    
    # Si usas batch, también hay que recortarlo
    batch = data.batch[subset_mask] if data.batch is not None else None
    
    # 4. Crear nuevo objeto data
    new_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    return new_data