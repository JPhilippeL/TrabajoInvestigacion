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
    graphexplanation_path, 
    gnnexplanation_path,
    mode = "alfa" 
):
    """
    Función orquestadora completa: 
    1. Procesa nombres y carga datos (SDF -> Grafo).
    2. Carga el modelo (Checkpoints).
    3. Carga tensores de explicación.
    4. Calcula curvas y genera gráfico.
    """
    
    # --- 1. Procesamiento de Strings y Nombres ---
    # Extraemos el nombre del modelo del path
    model_folder_name = model_path.split('/')[-1].split('.')[0]
    
    # Extraemos el ID y nombre de la molécula
    mol_id = os.path.basename(sdf_path).split('.')[0]

    # --- 3. Carga del Modelo ---
    try:
        model, device, _ = cargar_modelo(model_path)
        model.eval()
    except Exception as e:
        print(f"Error cargando el modelo desde {model_path}: {e}")
        return None
    
    # --- 2. Carga de Molécula y Conversión a Grafo ---
    if not os.path.exists(sdf_path):
        print(f"Error: No se encontró el archivo SDF en {sdf_path}")
        return None

    # Usamos RDKit para leer el SDF
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    
    if mol is None:
        print(f"Error: No se pudo leer la molécula del SDF.")
        return None

    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else mol_id
    
    # Convertimos a data object (asumiendo que esta función la tienes importada)
    data = mol_to_graph_data(mol)

    print(f"--- Iniciando Comparativa para {mol_name} (Modelo: {model_folder_name}) ---")

    # --- 4. Carga de Tensores de Importancia ---
    try:
        # Asumiendo que graphexplanation_path es 'path_pesos_mio'
        weights_mine = cargar_pesos_tensor(graphexplanation_path, device)
        # Asumiendo que gnnexplanation_path es 'path_pesos_gnn'
        weights_gnn = cargar_pesos_tensor(gnnexplanation_path, device)
    except FileNotFoundError as e:
        print(f"Error cargando archivos de pesos de explicación: {e}")
        return None

    # --- 5. Calcular Curva GraphExplainer (El tuyo) ---
    print("Calculando curva para GraphExplainer...")
    k_vals, fiab_mio = calcular_curvas_fidelity_general(model, data, weights_mine, device, mode)

    # --- 6. Calcular Curva GNNExplainer ---
    print("Calculando curva para GNNExplainer...")
    # Usamos _ en el primer retorno porque los k son idénticos
    _, fiab_gnn = calcular_curvas_fidelity_general(model, data, weights_gnn, device, mode)

    # --- 7. Generar Gráfico ---
    plot_path = guardar_plot_fidelity_comparativo(
        k_values=k_vals,
        fiab_my_explainer=fiab_mio,
        fiab_gnn_explainer=fiab_gnn,
        model_name=model_folder_name,
        mol_name=mol_name,
        mode = mode
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
    Genera un gráfico comparativo entre Tu Explainer y GNNExplainer.
    """
    
    # 1. Sanitizar nombre
    safe_mol_name = "".join([c for c in mol_name if c.isalnum() or c in (' ', '_', '-')]).strip()
    
    # 2. Configurar Rutas
    filename = f"COMPARATIVA_FIDELITY_{safe_mol_name}_{mode}.png"
    base_model_dir = os.path.join(RESULTADOS_DIR, model_name)
    fidelity_dir = os.path.join(base_model_dir, "Fidelity_Comparison")
    os.makedirs(fidelity_dir, exist_ok=True)
    full_save_path = os.path.join(fidelity_dir, filename)

    # 3. Calcular Áreas bajo la curva (AUC)
    # Cuanto mayor sea el AUC, mejor es el modelo identificando ruido (mantiene la predicción alta)
    auc_mine = np.trapezoid(fiab_my_explainer, k_values)
    auc_gnn = np.trapezoid(fiab_gnn_explainer, k_values)
    
    # 4. Plotting
    plt.figure(figsize=(10, 6))
    
    # --- Estilo para GraphExplainer (El tuyo) ---
    plt.plot(k_values, fiab_my_explainer, 
             marker='o', color='#1f77b4', linestyle='-', linewidth=2.5,
             label=f'GraphExplainer (AUC: {auc_mine:.2f})')
    
    # --- Estilo para GNNExplainer (Benchmark) ---
    plt.plot(k_values, fiab_gnn_explainer, 
             marker='x', color='#ff7f0e', linestyle='--', linewidth=2, alpha=0.9,
             label=f'GNNExplainer (AUC: {auc_gnn:.2f})')

    # Decoración
    plt.title(f"{mode} Robustness Comparison: {mol_name}", fontsize=13, fontweight='bold')
    plt.xlabel("K (Nodes removed - Least Important First)", fontsize=11)
    plt.ylabel("Prediction Stability (1.0 = Perfect)", fontsize=11)
    
    plt.ylim(-0.05, 1.05) 
    plt.axhline(1, color='gray', linestyle=':', alpha=0.5)
    
    # Relleno sutil para destacar la diferencia
    plt.fill_between(k_values, fiab_my_explainer, fiab_gnn_explainer, 
                     color='gray', alpha=0.1)

    plt.legend(fontsize=10, loc="lower left", frameon=True, fancybox=True, shadow=True)
    plt.grid(True, linestyle='-', alpha=0.3)
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
    
    # Si por casualidad se guardó un diccionario (legacy), intentamos sacar 'beta'
    if isinstance(weights, dict):
        if 'beta' in weights:
            weights = weights['beta']
        else:
            # Si es un dict pero no tiene beta, asumimos que es el tensor directo encapsulado
            weights = list(weights.values())[0]
            
    return weights

def calcular_curvas_fidelity_general(model, data, importance, device, mode = "beta", max_steps=15):
    """
    Calcula las curvas de fidelidad (Fidelity-) eliminando o perturbando información
    menos importante progresivamente.
    
    Args:
        mode (str): 'alfa' (features), 'beta' (nodos), 'gamma' (edge_attr), 'delta' (edges)
    """
    model.eval()
    data = data.to(device)
    
    # === 1. Determinar el límite de iteración según el modo ===
    if mode == 'alfa':
        total_elements = data.x.shape[1] # Num Features
    elif mode == 'beta':
        total_elements = data.x.shape[0] # Num Nodos
    elif mode == 'gamma':
        # Asumiendo que existen edge_attr
        total_elements = data.edge_attr.shape[1] if data.edge_attr is not None else 0
    elif mode == 'delta':
        total_elements = data.edge_index.shape[1] # Num Aristas
    else:
        raise ValueError(f"Modo {mode} no reconocido.")

    limit = total_elements
    print(total_elements)
    if max_steps is not None:
        limit = min(total_elements, max_steps)

    # === 2. Procesar Importancia ===
    # Aseguramos numpy aplanado y valor absoluto
    if torch.is_tensor(importance):
        imp = importance.detach().cpu().numpy().flatten()
    else:
        imp = np.array(importance).flatten()
        
    # Verificar que el tamaño de importance coincida con el elemento que estamos evaluando
    if len(imp) != total_elements:
        print(f"Advertencia: Longitud de importancia ({len(imp)}) != Elementos en modo {mode} ({total_elements}). Se recortará o fallará.")

    imp = np.abs(imp)
    
    # Orden ASCENDENTE (Fidelity-): quitamos primero lo que tiene MENOR importancia (ruido)
    sorted_indices = np.argsort(imp).copy()

    # === 3. Predicción Original ===
    with torch.no_grad():
        pred_original = model(data.x, data.edge_index, data.edge_attr, data.batch)
        val_orig = pred_original.item()

    fiab_list = []
    k_values = []

    # === 4. Bucle de Perturbación ===
    for k in range(limit + 1):
        k_values.append(k)
        
        # Índices acumulados a eliminar/perturbar hasta el paso k
        current_indices = sorted_indices[:k]

        if k == 0:
            data_minus = data
        else:
            # DESPACHADOR DE MODOS
            if mode == 'alfa':
                data_minus = ocultar_features_nodos(data, current_indices)
            
            elif mode == 'beta':
                # Tu función existente
                data_minus = eliminar_nodos_y_conexiones(data, current_indices)
            
            elif mode == 'gamma':
                # Pendiente para la siguiente interacción
                raise NotImplementedError("Modo Gamma aún no implementado")
                
            elif mode == 'delta':
                # Pendiente para la siguiente interacción
                raise NotImplementedError("Modo Delta aún no implementado")

        # === 5. Inferencia sobre grafo modificado ===
        with torch.no_grad():
            # Protección para Beta/Delta si el grafo queda vacío
            if data_minus.x.shape[0] == 0 or data_minus.edge_index.shape[1] == 0:
                val_minus = 0.0 # O comportamiento neutro definido
            else:
                pred_minus = model(data_minus.x, data_minus.edge_index, data_minus.edge_attr, data_minus.batch)
                val_minus = pred_minus.item()

        # Cálculo de métrica Fidelity
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