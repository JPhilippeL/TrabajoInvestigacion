import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from ui.utils import RESULTADOS_DIR

# Constante N: Número máximo de nodos a evaluar en la curva
MAX_NODES_FIDELITY = 15

def calcular_curvas_fiability(model, data, node_importance, device):
    """
    Calcula las curvas basándose en los nodos MENOS importantes (ruido).
    """
    model.eval()
    data = data.to(device)
    num_nodes = data.x.shape[0]
    
    # Aseguramos que importance sea numpy y aplanado
    if torch.is_tensor(node_importance):
        imp = node_importance.detach().cpu().numpy().flatten()
    else:
        imp = np.array(node_importance).flatten()

    # === ORDEN ASCENDENTE (De Menor a Mayor Importancia) ===
    # Los primeros índices son los MENOS importantes (ruido)
    sorted_indices = np.argsort(imp).copy() 
    # =======================================================

    # Predicción original
    with torch.no_grad():
        pred_original = model(data.x, data.edge_index, data.edge_attr, data.batch)
        val_orig = pred_original.item()

    fiab_plus_list = []
    fiab_minus_list = []
    k_values = []

    limit = min(num_nodes, MAX_NODES_FIDELITY)

    for k in range(limit + 1):
        k_values.append(k)
        current_k_indices = sorted_indices[:k]
        
        # --- Fidelity+ (Keep ONLY Low Imp) ---
        # Dejamos solo el ruido. Esperamos que la predicción falle (Bajo Score).
        mask_plus = torch.zeros(num_nodes, 1, device=device)
        mask_plus[current_k_indices] = 1.0
        
        data_plus = data.clone()
        data_plus.x = data.x * mask_plus
        
        with torch.no_grad():
            pred_plus = model(data_plus.x, data_plus.edge_index, data_plus.edge_attr, data_plus.batch)
            val_plus = pred_plus.item()
        
        diff_plus = abs(val_orig - val_plus)
        fiab_plus = np.exp(-diff_plus) 
        fiab_plus_list.append(fiab_plus)

        # --- Fidelity- (Remove ONLY Low Imp) ---
        # Quitamos el ruido. Esperamos que la predicción se mantenga (Alto Score).
        mask_minus = torch.ones(num_nodes, 1, device=device)
        mask_minus[current_k_indices] = 0.0 
        
        data_minus = data.clone()
        data_minus.x = data.x * mask_minus
        
        with torch.no_grad():
            pred_minus = model(data_minus.x, data_minus.edge_index, data_minus.edge_attr, data_minus.batch)
            val_minus = pred_minus.item()
            
        diff_minus = abs(val_orig - val_minus)
        fiab_minus = np.exp(-diff_minus)
        fiab_minus_list.append(fiab_minus)

    return k_values, fiab_plus_list, fiab_minus_list

def guardar_plot_fiability(k_values, fiab_plus, fiab_minus, model_name, mol_name, algo_name="Explainer"):
    """
    Genera el gráfico con los colores invertidos:
    - Fidelity- (Debe ser alto) -> VERDE
    - Fidelity+ (Debe ser bajo) -> ROJO
    """
    
    # 1. Sanitizar nombre
    safe_mol_name = "".join([c for c in mol_name if c.isalnum() or c in (' ', '_', '-')]).strip()
    safe_mol_name = safe_mol_name.replace(" ", "_")
    
    # 2. Nombre de archivo
    filename = f"FIABILITY_INV_{model_name}_{safe_mol_name}_{algo_name}.png"
    
    # 3. Directorios
    base_model_dir = os.path.join(RESULTADOS_DIR, model_name)
    fiability_dir = os.path.join(base_model_dir, "Fiability")
    os.makedirs(fiability_dir, exist_ok=True)
    
    full_save_path = os.path.join(fiability_dir, filename)

    # 4. AUC
    auc_plus = np.trapz(fiab_plus, k_values)
    auc_minus = np.trapz(fiab_minus, k_values)
    
    plt.figure(figsize=(10, 6))
    
    # Etiquetas
    label_plus = f'Fidelity+ (Keep ONLY Low Imp.)\nAUC: {auc_plus:.2f} (Ideal: Low)'
    label_minus = f'Fidelity- (Remove ONLY Low Imp.)\nAUC: {auc_minus:.2f} (Ideal: High)'

    # === COLORES MODIFICADOS ===
    # Fidelity+ -> Rojo (Queremos que baje)
    plt.plot(k_values, fiab_plus, marker='o', label=label_plus, color='red', linestyle='-', linewidth=2)
    
    # Fidelity- -> Verde (Queremos que se mantenga alto)
    plt.plot(k_values, fiab_minus, marker='x', label=label_minus, color='green', linestyle='--', linewidth=2)

    plt.title(f"Noise Robustness Analysis ({algo_name}): {mol_name}", fontsize=12, fontweight='bold')
    plt.xlabel("K (Number of Low Importance Nodes modified)", fontsize=10)
    plt.ylabel("Prediction Similarity $e^{-|Error|}$", fontsize=10)
    
    plt.ylim(-0.05, 1.05) 
    plt.axhline(1, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    # Rellenos (Match con los colores de las líneas)
    plt.fill_between(k_values, fiab_plus, color='red', alpha=0.1)
    plt.fill_between(k_values, fiab_minus, color='green', alpha=0.1)

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0,
               fontsize=9, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.xticks(k_values)
    plt.tight_layout()
    
    plt.savefig(full_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return full_save_path