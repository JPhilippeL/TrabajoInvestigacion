import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from rdkit import Chem
from ui.utils import RESULTADOS_DIR
from ML.data_processing import mol_to_graph_data
from ML.model_tester import cargar_modelo

# Constante N: Número máximo de nodos a evaluar en la curva
MAX_NODES_FIDELITY = 15

def generar_comparativa_fidelity(
    model_path, 
    sdf_path, 
    graphexp_weights_path, 
    gnnexp_weights_path, # Puede ser None
    mode = "delta" 
):
    """
    Función orquestadora completa.
    """
    
    # --- 1. Procesamiento de Strings y Nombres ---
    model_folder_name = model_path.split('/')[-1].split('.')[0]
    mol_id = os.path.basename(sdf_path).split('.')[0]

    # --- 2. Carga del Modelo ---
    try:
        model, device, _ = cargar_modelo(model_path)
        model.eval()
    except Exception as e:
        print(f"Error cargando el modelo desde {model_path}: {e}")
        return None
    
    # --- 3. Carga de Molécula y Conversión a Grafo ---
    if not os.path.exists(sdf_path):
        print(f"Error: No se encontró el archivo SDF en {sdf_path}")
        return None

    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    
    if mol is None:
        print(f"Error: No se pudo leer la molécula del SDF.")
        return None

    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else mol_id
    data = mol_to_graph_data(mol)

    print(f"--- Comparativa ({mode}) para {mol_name} ---")

    # --- 4. Carga de Tensores de Importancia ---
    
    # A) Carga GraphExplainer
    tensor_graphexp = cargar_pesos_tensor(graphexp_weights_path, device)

    # B) Carga GNNExplainer (Opcional)
    tensor_gnn = None
    if gnnexp_weights_path is not None or mode == "gamma":
        tensor_gnn = cargar_pesos_tensor(gnnexp_weights_path, device)
    else:
        print("Info: Se omite resultados GNNExplainer.")

    # --- 5. Calcular Curva GraphExplainer ---
    print(f"Calculando curva {mode} para GraphExplainer...")
    k_vals, fiab_graphexp = calcular_curvas_fidelity_general(model, data, tensor_graphexp, device, mode)

    # --- 6. Calcular Curva GNNExplainer---
    fiab_gnn = None
    if tensor_gnn is not None:
        print(f"Calculando curva {mode} para GNNExplainer...")
        try:
            _, fiab_gnn = calcular_curvas_fidelity_general(model, data, tensor_gnn, device, mode)
        except ValueError as e:
            print(f"Saltando GNNExplainer por incompatibilidad de dimensiones: {e}")
            fiab_gnn = None

    # --- 7. Generar Gráfico ---
    plot_path = guardar_plot_fidelity_comparativo(
        k_values=k_vals,
        fiab_my_explainer=fiab_graphexp,
        fiab_gnn_explainer=fiab_gnn,
        model_name=model_folder_name,
        mol_name=mol_name,
        mode=mode
    )
    
    return plot_path

def guardar_plot_fidelity_comparativo(
        k_values, 
        fiab_my_explainer, 
        fiab_gnn_explainer, 
        model_name, 
        mol_name,
        mode
    ):
    """
    Genera un gráfico comparativo. Si fiab_gnn_explainer es None,
    solo grafica GraphExplainer Explainer sin romper el código.
    """
    
    # 1. Sanitizar nombre
    safe_mol_name = "".join([c for c in mol_name if c.isalnum() or c in (' ', '_', '-')]).strip()
    
    # 2. Configurar Rutas
    filename = f"COMPARATIVA_FIDELITY_{safe_mol_name}_{mode}.png"
    base_model_dir = os.path.join(RESULTADOS_DIR, model_name) # Asegúrate que RESULTADOS_DIR es accesible
    fidelity_dir = os.path.join(base_model_dir, "Fidelity_Comparison")
    os.makedirs(fidelity_dir, exist_ok=True)
    full_save_path = os.path.join(fidelity_dir, filename)

    # 3. Calcular Áreas bajo la curva (AUC) - GraphExpplainer EXPLAINER
    auc_mine = np.trapezoid(fiab_my_explainer, k_values)

    # 4. Validar si existe GNNExplainer
    has_gnn = (fiab_gnn_explainer is not None) and (len(fiab_gnn_explainer) > 0)
    
    auc_gnn = 0.0
    if has_gnn:
        try:
            auc_gnn = np.trapezoid(fiab_gnn_explainer, k_values)
        except AttributeError:
            auc_gnn = np.trapz(fiab_gnn_explainer, k_values)
    
    # 5. Plotting
    plt.figure(figsize=(10, 6))
    
    # --- Estilo para GraphExplainer (El tuyo) ---
    plt.plot(k_values, fiab_my_explainer, 
             marker='o', color='#1f77b4', linestyle='-', linewidth=2.5,
             label=f'GraphExplainer (AUC: {auc_mine:.2f})')
    
    # --- Estilo para GNNExplainer (Solo si existe) ---
    if has_gnn:
        plt.plot(k_values, fiab_gnn_explainer, 
                 marker='x', color='#ff7f0e', linestyle='--', linewidth=2, alpha=0.9,
                 label=f'GNNExplainer (AUC: {auc_gnn:.2f})')
        
        # Relleno sutil para destacar la diferencia (solo si hay ambos)
        # Rellena donde 'graphexp' es mayor o menor que 'GNN'
        plt.fill_between(k_values, fiab_my_explainer, fiab_gnn_explainer, 
                         color='gray', alpha=0.1)

    # Decoración
    plt.title(f"{mode.capitalize()} Robustness Comparison: {mol_name}", fontsize=13, fontweight='bold')
    
    # Etiqueta X dinámica según el modo
    if mode == 'alfa':
        xlabel_text = "K (Node Features Perturbed - Least Important First)"
    elif mode == 'beta':
        xlabel_text = "K (Nodes Removed - Least Important First)"
    elif mode == 'gamma':
        xlabel_text = "K (Edge Features Perturbed - Least Important First)"
    elif mode == 'delta':
        xlabel_text = "K (Edges Removed - Least Important First)"
    else:
        xlabel_text = "K (Elements Perturbed)"

    plt.xlabel(xlabel_text, fontsize=11)
    plt.ylabel("Prediction Stability (1.0 = Perfect)", fontsize=11)
    
    plt.ylim(-0.05, 1.05) 
    plt.axhline(1, color='gray', linestyle=':', alpha=0.5)
    
    plt.legend(fontsize=10, loc="lower left", frameon=True, fancybox=True, shadow=True)
    plt.grid(True, linestyle='-', alpha=0.3)
    
    # Ajustar ticks del eje X si son pocos valores (para que se vean enteros)
    if len(k_values) < 20:
        plt.xticks(k_values)
        
    plt.tight_layout()
    
    plt.savefig(full_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico comparativo guardado en: {full_save_path}")
    return full_save_path

def cargar_pesos_tensor(path, device='cpu'):
    """
    Carga un tensor guardado en .pt y asegura que esté en el formato correcto.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {path}")
    
    weights = torch.load(path, map_location=device)
            
    return weights

def calcular_curvas_fidelity_general(model, data, importance, device, mode= "beta", max_steps=15):
    model.eval()
    data = data.to(device)
    
    # === 1. Determinar el límite y validar dimensiones ===
    if mode == 'alfa':
        total_elements = data.x.shape[1] # Num Features Nodos
    elif mode == 'beta':
        total_elements = data.x.shape[0] # Num Nodos
    elif mode == 'gamma':
        # Si no hay atributos de arista, no se puede calcular gamma
        if data.edge_attr is None:
            print("Aviso: Modo gamma solicitado pero data.edge_attr es None. Retornando vacío.")
            return [], []
        total_elements = data.edge_attr.shape[1] # Num Features Aristas
    elif mode == 'delta':
        total_elements = data.edge_index.shape[1] # Num Aristas
        max_steps*= 2
    else:
        raise ValueError(f"Modo {mode} no reconocido.")

    # Procesar Importancia
    if torch.is_tensor(importance):
        imp = importance.detach().cpu().numpy().flatten()
    else:
        imp = np.array(importance).flatten()

    # === CHECK DE SEGURIDAD ===
    if len(imp) != total_elements:
        raise ValueError(f"ERROR DE DIMENSIÓN: Modo '{mode}' espera {total_elements} elementos, "
                         f"pero el vector de importancia tiene longitud {len(imp)}. "
                         "Verifica que estás pasando el tensor correcto (alfa vs beta vs delta).")

    limit = total_elements
    if max_steps is not None:
        limit = min(total_elements, max_steps)

    imp = np.abs(imp)
    # Orden ascendente: primero eliminamos lo menos importante
    sorted_indices = np.argsort(imp).copy()

    # Predicción Original
    with torch.no_grad():
        pred_original = model(data.x, data.edge_index, data.edge_attr, data.batch)
        val_orig = pred_original.item()

    fiab_list = []
    k_values = []

    # === Bucle Principal ===
    for k in range(limit + 1):
        k_values.append(k)
        
        # Índices acumulados a perturbar
        current_indices = sorted_indices[:k]

        if k == 0:
            data_minus = data
        else:
            # === DESPACHADOR DE MODOS ===
            if mode == 'alfa':     # Features Nodos (Enmascarar con media)
                data_minus = ocultar_features_nodos(data, current_indices)
            
            elif mode == 'beta':   # Nodos (Eliminar nodo y conexiones)
                data_minus = eliminar_nodos_y_conexiones(data, current_indices)
            
            elif mode == 'gamma':  # Features Aristas (Enmascarar con media)
                data_minus = ocultar_features_aristas(data, current_indices)
                
            elif mode == 'delta':  # Aristas (Eliminar conexión)
                data_minus = eliminar_aristas_selectivas(data, current_indices)

        # Inferencia
        with torch.no_grad():
            # Check si el grafo quedó vacío o inválido
            if data_minus.x.shape[0] == 0: # Sin nodos
                val_minus = 0.0
            elif data_minus.edge_index.shape[1] == 0 and mode == 'delta': 
                # Si borramos todas las aristas, el GNN actúa solo sobre features de nodos aislados
                pred_minus = model(data_minus.x, data_minus.edge_index, data_minus.edge_attr, data_minus.batch)
                val_minus = pred_minus.item()
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

# ------- ALFA ---------
def ocultar_features_nodos(data, indices_features_a_ocultar):
    """
    MODO ALFA: Perturba las features indicadas reemplazándolas por 
    la media de dicha feature a través de todos los nodos.
    """
    # 1. Clonamos x para no modificar el original
    x_mod = data.x.clone()
    
    # 2. Calculamos la media por columna (feature)
    # x_mod tiene shape [Num_Nodos, Num_Features]
    feature_means = x_mod.mean(dim=0) # Shape: [Num_Features]
    
    # 3. Reemplazamos las columnas seleccionadas por su media
    # Para cada feature 'f' en la lista, asignamos feature_means[f] a todos los nodos
    if len(indices_features_a_ocultar) > 0:
        # Convertimos a tensor si es lista numpy
        idx_tensor = torch.tensor(indices_features_a_ocultar, device=data.x.device)
        x_mod[:, idx_tensor] = feature_means[idx_tensor]
        
    # 4. Retornamos nuevo objeto Data (mismo grafo, features perturbadas)
    new_data = Data(
        x=x_mod, 
        edge_index=data.edge_index, 
        edge_attr=data.edge_attr, 
        batch=data.batch
    )
    
    return new_data

# ------- BETA ---------
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

# ------- GAMMA ---------
def ocultar_features_aristas(data, indices_features_a_ocultar):
    """
    MODO GAMMA: Perturba las features de las aristas (edge_attr) reemplazándolas 
    por la media global de esa feature.
    """
    if data.edge_attr is None:
        return data

    # 1. Clonar edge_attr
    edge_attr_mod = data.edge_attr.clone()
    
    # 2. Calcular media por columna (feature de arista)
    # edge_attr shape: [Num_Edges, Num_Edge_Features]
    feature_means = edge_attr_mod.mean(dim=0) 
    
    # 3. Reemplazar columnas
    if len(indices_features_a_ocultar) > 0:
        idx_tensor = torch.tensor(indices_features_a_ocultar, device=data.x.device)
        # Protección de índices para evitar crash CUDA
        if idx_tensor.max() >= edge_attr_mod.shape[1]:
             raise ValueError(f"Índice {idx_tensor.max()} fuera de rango para {edge_attr_mod.shape[1]} features de arista.")
             
        edge_attr_mod[:, idx_tensor] = feature_means[idx_tensor]
        
    return Data(x=data.x, edge_index=data.edge_index, 
                edge_attr=edge_attr_mod, batch=data.batch)

# ------- DELTA ---------
def eliminar_aristas_selectivas(data, indices_aristas_a_eliminar):
    """
    MODO DELTA: Elimina aristas específicas (edges) basándose en su índice.
    No elimina nodos, solo desconecta.
    """
    num_edges = data.edge_index.shape[1]
    device = data.x.device
    
    # 1. Crear máscara de aristas a MANTENER
    # Inicialmente todas True
    edge_mask = torch.ones(num_edges, dtype=torch.bool, device=device)
    
    # Poner en False las que queremos eliminar
    if len(indices_aristas_a_eliminar) > 0:
        idx_tensor = torch.tensor(indices_aristas_a_eliminar, device=device)
        # Protección
        if idx_tensor.max() >= num_edges:
             raise ValueError(f"Índice de arista {idx_tensor.max()} fuera de rango (Total aristas: {num_edges})")
        edge_mask[idx_tensor] = False
        
    # 2. Filtrar edge_index y edge_attr
    new_edge_index = data.edge_index[:, edge_mask]
    
    new_edge_attr = None
    if data.edge_attr is not None:
        new_edge_attr = data.edge_attr[edge_mask]
        
    # 3. Retornar data (x y batch se mantienen igual)
    return Data(x=data.x, edge_index=new_edge_index, 
                edge_attr=new_edge_attr, batch=data.batch)