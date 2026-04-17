#explanation_visualization.py
from matplotlib import gridspec
import matplotlib.pyplot as plt
import torch
import matplotlib.colors as mcolors
import networkx as nx
import matplotlib.ticker as mticker
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

from ui.utils.constants import RESULTADOS_DIR, hybridization_types, periodic_elements
from graph_managment.sdf_converter import parse_sdf 

# --- FUNCIÓN MAESTRA ---
def guardar_dashboard_explicacion(
    # Datos del Grafo
    graph_obj,
    edge_index, 
    node_importance,
    edge_importance,
    
    # Datos de Heatmaps
    alfa_sorted,
    row_labels_alfa,
    gamma_sorted,
    row_labels_gamma,
    
    # Metadatos
    mol_name,
    target_name,
    real_val,
    pred_val,
    algo_name,
    model_name # <--- Nuevo argumento necesario para la carpeta
):
    """
    Genera, guarda y devuelve la ruta del dashboard de explicación.
    """
    
    # 1. Configuración de Dimensiones
    HEIGHT_GRAPH = 10.0   
    HEIGHT_PER_ROW = 0.4
    
    num_rows_alfa = alfa_sorted.shape[0] if alfa_sorted is not None else 0
    num_rows_gamma = gamma_sorted.shape[0] if gamma_sorted is not None else 0
    max_rows = max(num_rows_alfa, num_rows_gamma)
    height_heatmaps = max((max_rows * HEIGHT_PER_ROW) + 2.0, 4.0)
    total_height = HEIGHT_GRAPH + height_heatmaps

    # 2. Crear Figura
    fig = plt.figure(figsize=(16, total_height))
    
    main_title = generar_titulo_explicacion(mol_name, target_name, real_val, pred_val, algo_name)
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[HEIGHT_GRAPH, height_heatmaps], hspace=0.2)

    # --- PANEL 1: GRAFO ---
    ax_graph = fig.add_subplot(gs[0, :])
    node_idx_map = {str(n): int(n) for n in graph_obj.nodes}
    
    plot_graph_with_importance(
        graph_obj, 
        node_importance=node_importance, 
        edge_importance=edge_importance,
        edge_index=edge_index,
        ax=ax_graph, 
        node_idx_map=node_idx_map,
        cmap="plasma"
    )

    # --- PANEL 2: HEATMAP NODOS ---
    ax_alfa = fig.add_subplot(gs[1, 0])
    col_labels = [""]
    
    if alfa_sorted is not None and len(alfa_sorted) > 0:
        im_a, _ = heatmap(alfa_sorted, row_labels_alfa, col_labels, ax=ax_alfa, cmap="plasma", aspect='auto')
        annotate_heatmap(im_a, alfa_sorted, textcolors=("white", "black"))
        ax_alfa.set_title("Node Features", fontsize=14, pad=15)
    else:
        ax_alfa.text(0.5, 0.5, "No significant node features", ha='center')
        ax_alfa.axis('off')

    # --- PANEL 3: HEATMAP ARISTAS ---
    ax_gamma = fig.add_subplot(gs[1, 1])
    
    if gamma_sorted is not None and len(gamma_sorted) > 0:
        im_g, _ = heatmap(gamma_sorted, row_labels_gamma, col_labels, ax=ax_gamma, cmap="plasma", aspect='auto')
        annotate_heatmap(im_g, gamma_sorted, textcolors=("white", "black"))
        ax_gamma.set_title("Edge Features", fontsize=14, pad=15)
    else:
        ax_gamma.text(0.5, 0.5, "No edge features importance", ha='center')

    # 3. GESTIÓN DE DIRECTORIOS Y GUARDADO
    # Limpiamos nombres para evitar errores en sistema de archivos
    safe_mol_name = "".join([c for c in mol_name if c.isalnum() or c in (' ', '_', '-')]).strip()
    safe_algo_name = algo_name.lower().replace(" ", "")
    
    # --- CAMBIO AQUÍ ---
    # Construimos la ruta: Resultados / NombreModelo / NombreAlgoritmo
    # Asegúrate de que RESULTADOS_DIR esté definido o impórtalo
    output_dir = os.path.join(RESULTADOS_DIR, model_name, safe_algo_name)
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{model_name}_{safe_mol_name}_{safe_algo_name}.png"
    save_path = os.path.join(output_dir, filename)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def heatmap(data, row_labels, col_labels, ax, aspect='auto', **kwargs):
    """
    Crea un heatmap a partir de un array numpy y dos listas de etiquetas.
    Se ha añadido el parámetro 'aspect' para controlar la proporción de celdas.
    """
    if ax is None:
        ax = plt.gca()
    
    # Crear heatmap
    # aspect='auto' permite que las celdas se estiren para llenar el eje
    im = ax.imshow(data, aspect=aspect, **kwargs)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Feature Significance", rotation=-90, va="bottom")

    # Show all ticks and label them
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0, ha="center")
    
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mticker.StrMethodFormatter(valfmt)


    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_graph_with_importance(graph, node_importance, edge_importance=None, edge_index=None, ax=None, cmap="plasma", node_idx_map=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # --- 1. PROCESAMIENTO DE NODOS (Igual que antes) ---
    beta = np.array(node_importance, dtype=float)
    if node_idx_map is not None:
        beta_mapped = []
        for n in graph.nodes:
            idx = node_idx_map.get(str(n)) if str(n) in node_idx_map else node_idx_map.get(int(n))
            val = beta[idx] if idx is not None and idx < len(beta) else 0.0
            beta_mapped.append(val)
        beta = np.array(beta_mapped)

    pos = {n: graph.nodes[n]["pos"] for n in graph.nodes}
    
    # Normalización Nodos (Ahora con normalizar_por_norma como acordamos)
    # Si prefieres min_max en el grafo, cambia esta línea
    vmin_n, vmax_n = beta.min(), beta.max()
    norm_n = mcolors.Normalize(vmin=vmin_n, vmax=vmax_n) if vmax_n > 0 else mcolors.Normalize(vmin=0, vmax=1)
    cmap_obj = plt.cm.get_cmap(cmap)
    node_colors = [cmap_obj(norm_n(b)) for b in beta]

    # --- 2. PROCESAMIENTO DE ARISTAS POR TIPO ---
    
    # Preparamos diccionarios para agrupar las aristas por estilo visual
    # Esto es necesario porque nx.draw_networkx_edges dibuja todas las de la lista con el mismo estilo
    batches = {
        "solid":   {"edges": [], "colors": [], "widths": []},
        "dashed":  {"edges": [], "colors": [], "widths": []}, # Para Aromaticos
        "dotted":  {"edges": [], "colors": [], "widths": []}  # Opcional (ej. para puentes de H o interacciones)
    }

    if edge_importance is not None and edge_index is not None:
        delta = np.array(edge_importance, dtype=float)
        # Asumimos que delta YA viene normalizado (max=1)
        norm_e = mcolors.Normalize(vmin=0, vmax=1) 

        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.cpu().numpy()
            
        # Mapa de importancia (u, v) -> valor
        imp_dict = {}
        for i in range(len(delta)):
            u = int(edge_index[0, i])
            v = int(edge_index[1, i])
            val = delta[i]
            imp_dict[(u, v)] = val
            imp_dict[(v, u)] = val

        nx_edges = list(graph.edges())
        
        for u, v in nx_edges:
            # Recuperar indices reales
            u_real = node_idx_map.get(str(u)) if node_idx_map else int(u)
            v_real = node_idx_map.get(str(v)) if node_idx_map else int(v)

            # 1. Obtener Importancia
            val = imp_dict.get((u_real, v_real), 0.0)
            color = cmap_obj(norm_e(val))
            
            # 2. Obtener Tipo de Enlace del Grafo (NetworkX)
            # parse_sdf suele guardar esto en 'bond_type'
            # Puede venir como string ('SINGLE', 'AROMATIC') o float (1.0, 1.5, 2.0)
            edge_data = graph.get_edge_data(u, v)
            b_type = edge_data.get('bond_type', 'SINGLE') 
            
            # 3. Determinar Estilo y Grosor Base
            style = 'solid'
            final_width = 1.5 
            
            # Lógica de diferenciación
            b_type_str = str(b_type).upper()
            
            if 'AROMATIC' in b_type_str:
                style = 'dashed'
                final_width = 1.5
            elif 'DOUBLE' in b_type_str:
                style = 'solid'
                final_width = 2.5 # Doble enlace = más grueso
            elif 'TRIPLE' in b_type_str:
                style = 'solid' # O 'dotted' si prefieres diferenciar más
                final_width = 3.5 # Triple = muy grueso
            elif 'SINGLE' in b_type_str:
                # SINGLE
                style = 'solid'
                final_width = 1.5

            # Grosor final = Base + Importancia
            # final_width = final_width + (3.5 * val)

            # Guardar en el batch correspondiente
            if style in batches:
                batches[style]["edges"].append((u, v))
                batches[style]["colors"].append(color)
                batches[style]["widths"].append(final_width)

        # 4. DIBUJAR LOS BATCHES
        for style, data in batches.items():
            if data["edges"]:
                nx.draw_networkx_edges(
                    graph, pos, ax=ax,
                    edgelist=data["edges"],
                    edge_color=data["colors"],
                    width=data["widths"],
                    style=style,  # Aquí aplicamos solid/dashed/dotted
                    alpha=0.9
                )

    else:
        # Fallback si no hay info de LIME
        nx.draw_networkx_edges(graph, pos, ax=ax, width=1.5, alpha=0.4, edge_color="gray")
    
    # --- 3. DIBUJAR NODOS Y ETIQUETAS ---
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=300, edgecolors="black")
    labels = {n: graph.nodes[n].get("element", str(n)) for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels, font_size=9, ax=ax)
    
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm_n)
    sm.set_array([])
    
    # CAMBIO: Etiqueta actualizada
    plt.colorbar(sm, ax=ax, shrink=0.8, label="Node/Edge Significance")
    
    ax.axis("off")
    return ax

def obtener_info_real(target_data_path, mol_id):
    """
    Lee el archivo txt para obtener el nombre del target y el valor real.
    Asume formato "id valor" en cada línea.
    """
    if target_data_path is None or not os.path.exists(target_data_path):
        return "Unknown Target", None

    # Nombre del target basado en el nombre del archivo (sin .txt)
    target_name = os.path.basename(target_data_path).replace('.txt', '')
    
    real_val = None
    try:
        with open(target_data_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Buscamos que el ID coincida (primera columna)
                if len(parts) >= 2 and parts[0] == mol_id:
                    real_val = float(parts[1])
                    break
    except Exception as e:
        logger.error(f"Error leyendo archivo de targets: {e}")
        
    return target_name, real_val

def generar_titulo_explicacion(mol_name, target_name, real_val, pred_val, algo_name):
    """
    Genera el string del título formateado con el error si existe valor real.
    """
    if real_val is not None:
        try:
            val_float = float(real_val)
            error_val = abs(val_float - pred_val)
            real_str = f"{val_float:.4f}"
            error_str = f"{error_val:.4f}"
        except (ValueError, TypeError):
            real_str = str(real_val)
            error_str = "N/A"
    else:
        real_str = "N/A"
        error_str = "N/A"

    return (f"Compound: {mol_name}, {target_name}: {real_str},\n "
            f"Prediction: {pred_val:.4f}, Error: {error_str},\n {algo_name}")

def guardar_pesos(alfa, beta, gamma, delta, model_name, mol_name, algo_name):
    """
    Guarda los tensores de la explicación en subcarpetas específicas por variable.
    Ruta: Resultados/{model_name}/{variable}/{filename}
    """
    # 1. Sanitizar nombre de la molécula
    safe_mol_name = "".join([c for c in mol_name if c.isalnum() or c in (' ', '_', '-')]).strip()
    safe_mol_name = safe_mol_name.replace(" ", "_")
    
    # 2. Diccionario de tensores a guardar
    tensors_to_save = {
        'alfa': alfa,
        'beta': beta,
        'gamma': gamma,
        'delta': delta
    }
    
    saved_paths = []
    
    # 3. Iterar sobre cada variable para crear su propia carpeta y guardar
    for var_name, tensor in tensors_to_save.items():
        if tensor is not None:
            # CAMBIO PRINCIPAL:
            # Definir directorio específico: Resultados/model_name/alfa (o beta, etc.)
            specific_dir = os.path.join(RESULTADOS_DIR, model_name, var_name)
            os.makedirs(specific_dir, exist_ok=True)
            
            # Construir nombre del archivo
            filename = f"{var_name}_{algo_name}_{model_name}_{safe_mol_name}.pt"
            full_path = os.path.join(specific_dir, filename)
            
            # Guardamos en CPU
            torch.save(tensor.detach().cpu(), full_path)
            saved_paths.append(full_path)
            
    # Retornamos la ruta base del modelo para referencia
    base_model_dir = os.path.join(RESULTADOS_DIR, model_name)
    print(f"--- Pesos guardados separadamente en subcarpetas de: {base_model_dir} ---")
    
    return saved_paths

def get_feature_names_embedding():
    return [
        "Atom Symbol", 
        "Hybridization", 
        "Degree", 
        "Total Hs", 
        "Is Aromatic", 
        "Formal Charge", 
        "Gasteiger Charge", 
        "Is Donor", 
        "Is Acceptor"
    ]


def get_features_names_onehot():
    feature_names = []

    # Tipo de atomo one hot
    feature_names+= [f"Atom_{atom}" for atom in periodic_elements]

    # Features escalares
    feature_names.append("Degree")
    feature_names.append("Total Hs")
    feature_names.append("Is Aromatic")
    feature_names.append("Formal Charge")
    feature_names.append("Gasteiger Charge")
    feature_names.append("Is Donor")
    feature_names.append("Is Acceptor")

    # Hybridizacion
    feature_names+= [f"Hybrid_{h}" for h in hybridization_types]

    return feature_names

def normalizar_por_norma(arr):
    """
    Normaliza el vector usando la Norma L2 (Euclidiana).
    - La suma de los cuadrados de los elementos será 1.
    - Captura mejor la distribución de energía del vector.
    """
    if arr is None or arr.size == 0: 
        return arr
    
    # Calcular la norma L2 (raíz de la suma de cuadrados)
    norma = np.linalg.norm(arr)
    
    if norma == 0:
        return np.zeros_like(arr)
        
    return arr / norma

def tensor_to_abs_numpy(tensor):
    """Convierte tensor a numpy, toma valor absoluto."""
    if tensor is None: return None
    return np.abs(tensor.detach().cpu().numpy().reshape(-1, 1))

def procesar_features_ordenadas(importance_tensor, feature_names, input_data=None):
    """
    Procesa features para Heatmaps usando Max Scaling.
    MODIFICADO: Ya no filtra features que valen 0, porque en modo Embedding 
    el 0 es una categoría válida (ej. Single Bond o Carbono).
    """
    if importance_tensor is None:
        return None, []
    
    # 1. Obtener magnitudes crudas (Valor Absoluto)
    raw_imp = tensor_to_abs_numpy(importance_tensor)
    
    # 2. SIN FILTRADO (Corrección para Embedding)
    # En embedding, siempre queremos ver todas las features (9 nodos, 2 aristas).
    # Asumimos que todas existen.
    
    # Si quieres, puedes mantener un filtrado de seguridad por dimensión,
    # pero NO por contenido igual a cero.
    filtered_imp = raw_imp
    
    # Aseguramos que feature_names sea numpy array para indexado cómodo si hiciera falta
    filtered_names = np.array(feature_names)
    
    # Safety check de dimensiones
    if len(filtered_names) != len(filtered_imp):
        logger.warning(f"Dimension mismatch in processing: Names {len(filtered_names)} vs Imp {len(filtered_imp)}")
        # Cortamos al mínimo común para evitar crash
        min_len = min(len(filtered_names), len(filtered_imp))
        filtered_names = filtered_names[:min_len]
        filtered_imp = filtered_imp[:min_len]

    if len(filtered_imp) == 0:
        return np.array([]), []

    # 3. Ordenar (Mayor a menor)
    sort_idx = np.argsort(filtered_imp.flatten())[::-1]
    
    sorted_imp = filtered_imp[sort_idx]
    sorted_names = filtered_names[sort_idx]
    
    # 4. NORMALIZAR CON norma
    final_imp = normalizar_por_norma(sorted_imp)
    
    return final_imp, sorted_names.tolist()

def procesar_features_onehot(importance_tensor, feature_names, input_data):
    """
    Procesa features para One-Hot Encoding:
    1. FILTRA: Elimina features que no existen en la muestra (columna == 0).
    2. ORDENA: Por importancia absoluta.
    3. NORMALIZA: Usando Norma L2.
    
    Args:
        importance_tensor (Tensor): Pesos aprendidos (alfa o gamma).
        feature_names (list): Lista de nombres de las features.
        input_data (Tensor): Matriz de datos real de la muestra [N, Features].
                             Necesario para saber qué columnas son 0.
    """
    if importance_tensor is None or input_data is None:
        return np.array([]), []
    
    # 1. Obtener magnitudes crudas (Valor Absoluto) y convertir a Numpy
    # flatten() es importante para asegurar que sea un vector 1D
    raw_imp = tensor_to_abs_numpy(importance_tensor).flatten()
    
    # Convertir feature_names a numpy array para poder indexar con máscara booleana
    names_np = np.array(feature_names)
    
    # --- VALIDACIÓN DE DIMENSIONES ---
    num_features_data = input_data.shape[1]
    num_features_names = len(names_np)
    num_features_imp = len(raw_imp)
    
    # Ajustar al mínimo común denominador para evitar crashes
    min_len = min(num_features_data, num_features_names, num_features_imp)
    
    if num_features_data != num_features_names or num_features_data != num_features_imp:
        logger.warning(f"Mismatch dimensiones OneHot: Data={num_features_data}, Names={num_features_names}, Imp={num_features_imp}. Recortando a {min_len}.")
        
    # Recortamos todos al tamaño seguro
    input_data = input_data[:, :min_len]
    names_np = names_np[:min_len]
    raw_imp = raw_imp[:min_len]

    # 2. FILTRADO: Detectar columnas activas en la data
    # (input_data != 0).any(dim=0) devuelve True si hay al menos un 1 en esa columna
    active_mask = (input_data != 0).any(dim=0).cpu().numpy()
    
    # Aplicar máscara
    filtered_imp = raw_imp[active_mask]
    filtered_names = names_np[active_mask]
    
    if len(filtered_imp) == 0:
        return np.empty((0, 1)), []

    # 3. ORDENAR (Mayor a menor)
    sort_idx = np.argsort(filtered_imp)[::-1]
    
    sorted_imp = filtered_imp[sort_idx]
    sorted_names = filtered_names[sort_idx]
    
    # 4. NORMALIZAR CON NORMA L2 (Sobre las features que quedaron)
    final_imp = normalizar_por_norma(sorted_imp)
    
    # === CORRECCIÓN AQUÍ ===
    # Convertimos el array 1D (9,) de vuelta a una matriz columna 2D (9, 1)
    return final_imp.reshape(-1, 1), sorted_names.tolist()

# ====================================================================
# 1. LOGICA DE EXTRACCIÓN (HELPER)
# ====================================================================
def extraer_pesos_torchexplainers(explanation):
    """
    Convierte la salida compleja de GNNExplainer en los 4 tensores base (Raw).
    Mantiene el orden original de los nodos (índices).
    """
    node_mask = explanation.node_mask
    edge_mask = explanation.edge_mask

    # --- ALFA (Features Importancia Global) ---
    # Promedio por columna (features). Mantiene dimensión [Num_Features]
    if node_mask.ndim > 1:
        alfa_raw = node_mask.mean(dim=0) 
    else:
        alfa_raw = node_mask

    # --- BETA (Nodos Importancia Estructural) ---
    # Promedio por fila (nodos). Mantiene dimensión [Num_Nodos]
    # CRÍTICO: Mantiene el orden de índices para Fidelity
    if node_mask.ndim > 1:
        beta_raw = node_mask.mean(dim=1)
    else:
        beta_raw = node_mask

    # --- DELTA (Aristas) ---
    delta_raw = edge_mask if edge_mask is not None else None
    
    # --- GAMMA (Features Arista) ---
    # GNNExplainer estándar no suele devolver feature mask para aristas
    gamma_raw = None 

    return alfa_raw, beta_raw, gamma_raw, delta_raw


# ====================================================================
# 2. PIPELINE DE ANÁLISIS Y VISUALIZACIÓN
# ====================================================================
def pipeline_visualizacion_torchexplainers(
    alfa_raw, beta_raw, delta_raw, gamma_raw,
    edge_index, sdf_path, model, data, device,
    mol_name, target_name, real_val, pred_val, 
    model_name, algo_name="GNNExplainer"
):
    """
    Toma los pesos CRUDOS y coordina:
    1. Cálculo de Fidelity (con orden original).
    2. Procesamiento para Heatmaps (ordenar y etiquetar).
    3. Generación del Dashboard.
    """
    
    # A. PREPARACIÓN DE DATOS BASE
    graph_obj = parse_sdf(sdf_path)
    
    # Convertimos a numpy normalizado (Norma) para pintar y para el threshold de fidelity
    beta_np = normalizar_por_norma(tensor_to_abs_numpy(beta_raw))
    delta_normalized = normalizar_por_norma(tensor_to_abs_numpy(delta_raw)) if delta_raw is not None else np.array([])

    # ---------------------------------------------------------
    # B. CÁLCULO DE FIDELITY (Sobre datos alineados con grafos)
    # ---------------------------------------------------------
    # if model is not None and data is not None:
    #     try:
    #         logger.info("Calculando curvas de fidelity...")
    #         k_vals, fiab_minus = calcular_curvas_fidelity_general(
    #             model, data, beta_np, device
    #         )
    #         guardar_plot_fidelity(
    #             k_values=k_vals, fiab_minus=fiab_minus, 
    #             model_name=model_name, mol_name=mol_name, algo_name=algo_name
    #         )
    #     except Exception as e:
    #         logger.error(f"Error calculando fidelity: {e}")

    # ---------------------------------------------------------
    # C. PROCESAMIENTO PARA HEATMAPS (Aquí sí alteramos el orden)
    # ---------------------------------------------------------
    # Solo necesario para ALFA y GAMMA que llevan etiquetas de texto
    feature_names = get_feature_names_embedding()
    
    # ALFA
    if alfa_raw is not None:
        alfa_sorted, row_labels_alfa = procesar_features_ordenadas(
            alfa_raw, feature_names, data.x
        )
    else:
        alfa_sorted, row_labels_alfa = None, []

    # GAMMA
    if gamma_raw is not None:
        gamma_sorted, row_labels_gamma = procesar_features_ordenadas(
            gamma_raw, ["Bond Type", "Dist"], data.edge_attr
        )
    else:
        gamma_sorted, row_labels_gamma = None, []

    # ---------------------------------------------------------
    # D. GENERAR DASHBOARD FINAL
    # ---------------------------------------------------------
    plot_path = guardar_dashboard_explicacion(
        graph_obj=graph_obj,
        edge_index=edge_index,
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
    
    return plot_path