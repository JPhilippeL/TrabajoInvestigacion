# model_explainer.py
from ML.model_tester import cargar_modelo, predecir_molecula
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import matplotlib.ticker as mticker
import numpy as np
from ui.utils import RESULTADOS_DIR, periodic_elements, hybridization_types, N_BOND_TYPES
import os
import sys
import logging
from ML.data_processing import mol_to_graph_data_obj, mol_to_graph_data_obj_embedding
from rdkit import Chem
from core.sdf_converter import parse_sdf
import matplotlib.gridspec as gridspec

SIGMADIST = 1
MININICIAL = sys.float_info.max
logger = logging.getLogger(__name__)

def onehot_to_indices(data):
    """
    Convierte las features de nodos one-hot a indices que pueda usar EmbeddingEncoder.
    data.x debe tener:
        [one-hot atom | grado | numH | aromatic | one-hot hybrid]
    """
    x = data.x.clone()
    num_atoms = len(periodic_elements)
    num_hybrids = len(hybridization_types)

    # Atom index
    atom_onehot = x[:, :num_atoms]
    atom_idx = atom_onehot.argmax(dim=1, keepdim=True)

    # Continuos
    cont_features = x[:, num_atoms:-num_hybrids]

    # Hybrid index
    hybrid_onehot = x[:, -num_hybrids:]
    hybrid_idx = hybrid_onehot.argmax(dim=1, keepdim=True)

    # Concatenar: [atom_idx, hybrid_idx, cont_features]
    x_new = torch.cat([atom_idx, hybrid_idx, cont_features], dim=1)
    data_new = data.clone()
    data_new.x = x_new

    # --- 2. CONVERSIÓN DE ARISTAS (El error oculto) ---
    # Estructura entrada: [BondOneHot (4 cols) | Distancia]
    # Estructura salida:  [BondIdx (1 col) | Distancia]
    if data_new.edge_attr is not None and data_new.edge_attr.shape[1] > 2:
        edge_attr = data_new.edge_attr
        # Asumimos que la distancia es la ÚLTIMA columna
        dist = edge_attr[:, -1].unsqueeze(1)
        
        # El one-hot son todas las columnas menos la última
        bond_onehot = edge_attr[:, :-1]
        bond_idx = bond_onehot.argmax(dim=1, keepdim=True).float()
        
        data_new.edge_attr = torch.cat([bond_idx, dist], dim=1)

    return data_new


# Función para dada una muestra (x), genere una muestra perturbada (Z)
# Dado un vector binario de las características a perturbar
# feature_mask espera: [Atom, Degree, Arom, Hybrid, BondType, BondDist]
def perturb_features_sample(data, feature_mask, noise_level=0.05):
    data_new = data.clone()
    
    # ==========================================
    # 1. PERTURBACIÓN DE NODOS (Igual que antes)
    # ==========================================
    x = data_new.x
    start_atom, end_atom = 0, len(periodic_elements)
    start_degree, end_degree = end_atom, end_atom+1
    start_hs, end_hs = end_degree, end_degree+1
    aromatic_idx = end_hs
    start_hybrid, end_hybrid = aromatic_idx+1, aromatic_idx+1+len(hybridization_types)
    
    num_nodes = x.shape[0]

    # Rellenar máscara si falta
    if len(feature_mask) < 6:
        feature_mask = list(feature_mask) + [False] * (6 - len(feature_mask))

    for i in range(num_nodes):
        if feature_mask[0]:  # Átomo
            onehot = x[i, start_atom:end_atom]
            onehot[:] = 0
            new_idx = torch.randint(0, len(periodic_elements), (1,))
            onehot[new_idx] = 1
        if feature_mask[1]:  # Grado
            noise = noise_level * torch.randn_like(x[i, start_degree:end_hs])
            x[i, start_degree:end_hs] += noise
            # Asegurarse q no se pasa de 1
            x[i, start_degree:end_hs] = torch.clamp(x[i, start_degree:end_hs], 0.0, 1.0)
        if feature_mask[2]:  # Aromaticidad
            x[i, aromatic_idx] = 1 - x[i, aromatic_idx]
        if feature_mask[3]:  # Hibridación
            onehot = x[i, start_hybrid:end_hybrid]
            onehot[:] = 0
            new_idx = torch.randint(0, len(hybridization_types), (1,))
            onehot[new_idx] = 1

    data_new.x = x

    # ==========================================
    # 2. PERTURBACIÓN DE ARISTAS (SIMÉTRICA)
    # ==========================================
    if data_new.edge_attr is not None:
        edge_attr = data_new.edge_attr
        edge_index = data_new.edge_index
        num_edges = edge_attr.shape[0]
        
        # 1. Crear mapa para buscar el reverso rápido: (u, v) -> index
        # Esto nos permite encontrar dónde está B->A sabiendo A->B
        edge_map = {}
        for i in range(num_edges):
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()
            edge_map[(u, v)] = i

        # Definir slices
        num_bond_cols = N_BOND_TYPES 
        slice_bond = slice(0, num_bond_cols)
        idx_dist = -1

        for i in range(num_edges):
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()

            # --- TRUCO DE SIMETRÍA ---
            # Solo calculamos el ruido si u < v.
            # Si u > v, significa que ya procesamos este par cuando estábamos en (v, u)
            # y ya copiamos los datos, así que saltamos.
            if u > v:
                continue

            # --- APLICAR PERTURBACIÓN AL ENLACE 'i' ---
            modified = False
            
            # Perturbar Tipo de Enlace
            if feature_mask[4]:
                onehot_bond = edge_attr[i, slice_bond]
                onehot_bond[:] = 0
                new_bond_idx = torch.randint(0, num_bond_cols, (1,))
                onehot_bond[new_bond_idx] = 1
                modified = True
            
            # Perturbar Distancia
            if feature_mask[5]: # Distancia
                noise = noise_level * torch.randn_like(edge_attr[i, idx_dist])
                edge_attr[i, idx_dist] += noise
                # AGREGAR ESTO:
                edge_attr[i, idx_dist] = torch.clamp(edge_attr[i, idx_dist], 0.0, 1.0)

            # --- SINCRONIZAR CON EL ENLACE REVERSO ---
            if modified:
                # Buscamos el índice del enlace (v, u)
                if (v, u) in edge_map:
                    sym_idx = edge_map[(v, u)]
                    # Copiamos exactamente los mismos valores
                    edge_attr[sym_idx] = edge_attr[i].clone()

        data_new.edge_attr = edge_attr

    return data_new

# Función para generar múltiples muestras perturbadas
# La distribución de las muestras aleatorias tienen que seguir una distribucion gaussiana
def generate_perturbed_samples(data, feature_mask, num_samples=50, noise_level=0.05):
    perturbed_samples = []
    for i in range(num_samples):
        perturbed_sample = perturb_features_sample(data, feature_mask, noise_level)
        perturbed_samples.append(perturbed_sample)
    return perturbed_samples

import torch

def graph_feature_distance_list(x, z_list, metric='euclidean'):

    # distancia promedio usando las features de nodos

    distances = []
    for z in z_list:
        if x.x.shape != z.x.shape:
            raise ValueError("x y z deben tener la misma forma de features")

        diff = x.x - z.x

        if metric == 'euclidean':
            dist = torch.norm(diff, dim=1).mean()
        elif metric == 'cosine':
            sim = torch.nn.functional.cosine_similarity(x.x, z.x, dim=1)
            dist = 1 - sim.mean()
        else:
            raise ValueError(f"Métrica '{metric}' no soportada.")
        
        distances.append(dist.item())
    
    return distances


def embedding_distance_list(model, x, z_list, edge_attr_list=None, batch=None, device='cpu'):
    """
    Calcula distancias euclidianas entre embeddings de x y cada z en z_list.
    """
    model.eval()
    x, z_list = x.to(device), [z.to(device) for z in z_list]
    batch = torch.zeros(x.num_nodes, dtype=torch.long, device=device) if batch is None else batch

    with torch.no_grad():
        emb_x = model.get_embedding(x.x, x.edge_index, getattr(x, 'edge_attr', None), batch)
        distances = []
        for i, z in enumerate(z_list):
            z_batch = torch.zeros(z.num_nodes, dtype=torch.long, device=device)
            emb_z = model.get_embedding(z.x, z.edge_index, getattr(z, 'edge_attr', None), z_batch)
            distances.append(torch.norm(emb_x - emb_z).item())

    return distances

def obtener_lime(checkpoint_path, sdf_path, feature_mask, num_samples=50, noise_level=0.05, device='cpu'):
    
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    muestra = mol_to_graph_data_obj(mol)
    muestra_embedding = mol_to_graph_data_obj_embedding(mol)

    # Mapear node index -> atom idx
    node_to_atomidx = {i: atom.GetIdx() for i, atom in enumerate(mol.GetAtoms())}

    # Imprimir para verificación
    # for i in range(muestra.num_nodes):
    #    atom = mol.GetAtomWithIdx(node_to_atomidx[i])
    #    logger.info(f"Node {i} -> AtomIdx {atom.GetIdx()} ({atom.GetSymbol()})")
    
    # Generar muestras perturbadas
    perturbed_samples = generate_perturbed_samples(muestra, feature_mask, num_samples, noise_level)

    # Obtener modelo
    model, device, target_name = cargar_modelo(checkpoint_path)

    # Nombre y prediccion para el plot
    #prediccion_og = predecir_molecula(model, muestra_embedding, device)

    muestra_for_model = onehot_to_indices(muestra.to(device))
    prediccion_original = predecir_molecula(model, muestra_for_model, device)

    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else os.path.basename(sdf_path).split('.')[0]

    # Predecir las muestras perturbadas
    predicciones_perturbadas = []
    for perturbed in perturbed_samples:
        perturbed = perturbed.to(device)
        perturbed_for_model = onehot_to_indices(perturbed)  # <-- aquí el puente
        pred = predecir_molecula(model, perturbed_for_model, device)
        predicciones_perturbadas.append(pred)

    # Convertir a tensor [num_samples,1]
    predicciones_perturbadas = torch.tensor(predicciones_perturbadas, dtype=torch.float, device=device).unsqueeze(1)

    # Calcular distancias entre la muestra original y las perturbaciones
    feature_distances = graph_feature_distance_list(muestra, perturbed_samples, metric='euclidean')
    #feature_distances = embedding_distance_list(model, data_sample, perturbed_samples, device=device)

    # Precalcular todos los E_z^T (cada uno [d, N_z])
    E_t_list = [data_z.x.t().to(device) for data_z in perturbed_samples]

    # Lo mismo con los edges
    A_t_list = []
    for data_z in perturbed_samples:
        if data_z.edge_attr is not None:
            # Transponemos para que quede [Features x NumeroEdges]
            A_t_list.append(data_z.edge_attr.t().to(device))

    # --------------- HACERLO CON OPTIMIZACION DE PYTORCH ----------------------
    alfa, beta, gamma, delta, loss = obtener_argmin(feature_distances, predicciones_perturbadas, E_t_list, A_t_list, 0.01)
    # Verificar que aprendimos algo distinto de cero
    print(f"Max Alfa: {alfa.max().item():.4f}, Min Alfa: {alfa.min().item():.4f}")
    print(f"Max Beta: {beta.max().item():.4f}, Min Beta: {beta.min().item():.4f}")
    
    # --- Convertir y Normalizar (0 a 1) ---
    # Esto aplica: Tensor -> Numpy -> Abs -> MinMaxScaling
    alfa_np = procesar_y_normalizar(alfa)  # Importancia features nodo
    beta_np = procesar_y_normalizar(beta)  # Importancia nodos (grafo)

    # Si usas edges, normalizamos, si no, dejamos arrays vacíos/nulos
    if gamma is not None and gamma.numel() > 0:
        gamma_np = procesar_y_normalizar(gamma)
        delta_np = procesar_y_normalizar(delta)
    else:
        gamma_np = np.array([])
        delta_np = np.array([])

    # --- Etiquetas ---
    feature_names = get_feature_names(periodic_elements, hybridization_types)
    alfa_np, row_labels_alfa = filtrar_features_presentes(alfa_np, feature_names, muestra.x)
    col_labels_alfa = [""]

    row_labels_beta = [f"Node {i}" for i in range(len(beta_np))]
    col_labels_beta = [""]

    # ==========================================================================
    # CÁLCULO DE TAMAÑO DINÁMICO
    # ==========================================================================
    # Definimos una altura fija para el grafo (grande) y una altura por fila para las tablas
    HEIGHT_GRAPH = 10.0   # Pulgadas para el grafo
    HEIGHT_PER_ROW = 0.4  # Pulgadas por cada fila de la tabla (ajusta si quieres más espacio)
    
    num_rows_alfa = alfa_np.shape[0]
    num_rows_beta = beta_np.shape[0]
    max_rows = max(num_rows_alfa, num_rows_beta)
    
    # Altura necesaria para los heatmaps (+ un margen para títulos)
    height_heatmaps = (max_rows * HEIGHT_PER_ROW) + 2.0 
    
    # Altura total de la figura
    total_height = HEIGHT_GRAPH + height_heatmaps

    fig = plt.figure(figsize=(16, total_height))
    
    main_title = f"LIME Explanation for: **{mol_name}**\nModel Prediction: **{prediccion_original:.4f}** ({target_name})"
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.99) # y=0.99 para ponerlo arriba del todo

    # GridSpec: 2 Filas. 
    # Fila 0: Grafo (ocupa HEIGHT_GRAPH)
    # Fila 1: Heatmaps (ocupa height_heatmaps)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[HEIGHT_GRAPH, height_heatmaps], hspace=0.2)

    # 1. GRAFO (Fila 0, ocupa todas las columnas)
    ax_graph = fig.add_subplot(gs[0, :])
    
    # 2. HEATMAPS (Fila 1, columnas separadas)
    ax_alfa = fig.add_subplot(gs[1, 0])
    ax_beta = fig.add_subplot(gs[1, 1])

    # --- Plot Grafo ---
    graph = parse_sdf(sdf_path)
    node_idx_map = {str(atom.GetIdx()): atom.GetIdx() for atom in mol.GetAtoms()}
    
    plot_graph_with_importance(
        graph, 
        node_importance=beta_np.flatten(), 
        edge_importance=delta_np.flatten(), 
        edge_index=muestra.edge_index,
        ax=ax_graph, 
        node_idx_map=node_idx_map,
        cmap="plasma"
    )
    # --- Plot Heatmaps ---
    # Pasamos aspect='auto' para que use la altura física que hemos calculado
    im_a, _ = heatmap(alfa_np, row_labels_alfa, col_labels_alfa, ax=ax_alfa, cmap="plasma", aspect='auto')
    annotate_heatmap(im_a, alfa_np, textcolors=("white", "black"))
    ax_alfa.set_title("Feature Importance (Alpha)", fontsize=14)

    im_b, _ = heatmap(beta_np, row_labels_beta, col_labels_beta, ax=ax_beta, cmap="plasma", aspect='auto')
    annotate_heatmap(im_b, beta_np, textcolors=("white", "black"))
    ax_beta.set_title("Node Importance (Beta)", fontsize=14)

    # plt.tight_layout(rect=[0, 0, 1, 0.98]) # Ajuste para no tapar el título principal

    # Guardar
    model_name = checkpoint_path.split('/')[-1].split('.')[0]
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    model_results_dir = os.path.join(RESULTADOS_DIR, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    plotfilename = os.path.join(model_results_dir, f"{model_name}_for_{mol_name}_lime.png")

    plt.savefig(plotfilename)
    plt.close(fig)

    # Return de la imagen guardada
    return plotfilename

def obtener_argmin(feature_distances, predicciones_perturbadas, 
                   E_t_list, A_t_list, 
                   lr=0.05, 
                   epochs=2000, 
                   verbose=True):
    
    device = predicciones_perturbadas.device
    
    # --- DIAGNÓSTICO DE VARIANZA ---
    # Si esto es 0 o muy bajo, LIME no puede aprender nada porque el modelo
    # predice lo mismo para todas las perturbaciones.
    std_preds = predicciones_perturbadas.std().item()
    if verbose:
        logger.info(f"Desviación estándar de las predicciones: {std_preds:.6f}")
        if std_preds < 1e-5:
            logger.warning("¡CUIDADO! El modelo predice casi lo mismo para todas las muestras. Aumenta el noise_level.")

    # 1. Stack
    E_stack = torch.stack(E_t_list).to(device) 
    has_edges = len(A_t_list) > 0 and A_t_list[0] is not None
    if has_edges:
        A_stack = torch.stack(A_t_list).to(device)
        d_edges, M_edges = A_stack.shape[1], A_stack.shape[2]
    else:
        d_edges, M_edges = 1, 1

    num_samples, d_nodes, N_nodes = E_stack.shape

    # 2. CAMBIO CLAVE: ELIMINAMOS LA NORMALIZACIÓN AGRESIVA AQUÍ
    # Dejamos que los pesos aprendan su magnitud natural.
    # La normalización visual la haces tú después con 'procesar_y_normalizar'.
    # Para Nodos
    scale_nodes = 1.0 / math.sqrt(d_nodes * N_nodes)
    
    # Para Edges
    if has_edges:
        # Evitamos división por cero si N_edges es muy pequeño
        denom_edges = d_edges * M_edges
        scale_edges = 1.0 / math.sqrt(denom_edges) if denom_edges > 0 else 1.0
    else:
        scale_edges = 0.0

    # 3. Inicialización (Un poco más grande para ayudar al gradiente)
    alfa = nn.Parameter(torch.randn(d_nodes, 1, device=device) * 0.1)
    beta = nn.Parameter(torch.randn(N_nodes, 1, device=device) * 0.1)
    
    mean_pred = predicciones_perturbadas.mean().item()
    mu = nn.Parameter(torch.tensor([mean_pred], device=device, dtype=torch.float))

    params = [alfa, beta, mu]
    
    gamma = None
    delta = None

    if has_edges:
        gamma = nn.Parameter(torch.randn(d_edges, 1, device=device) * 0.1)
        delta = nn.Parameter(torch.randn(M_edges, 1, device=device) * 0.1)
        params.extend([gamma, delta])
    
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

    # ... dentro de obtener_argmin ...

    dists = torch.tensor(feature_distances, dtype=torch.float, device=device).view(-1, 1)
    
    # --- CORRECCIÓN CRÍTICA AQUÍ ---
    # NO usar .std() porque tus distancias son muy similares entre sí.
    # Usamos la media para asegurar que el kernel cubra tus datos.
    # Un buen heurístico es sigma = media * 0.75
    dist_mean = dists.mean()
    sigma = dist_mean if dist_mean > 0 else 1.0
    
    # Calculamos pesos.
    # Nota: Eliminamos el cuadrado del denominador dentro del exp para suavizar,
    # o usamos 2*sigma^2 si sigma es suficientemente grande.
    # Probemos esta forma robusta:
    weights = torch.exp(-(dists**2) / (2 * sigma**2))
    
    # IMPORTANTE: Re-escalar los pesos para que sumen 'num_samples'.
    # Esto evita que el Loss sea pequeñísimo (0.0001) y que los gradientes mueran.
    weights = weights / weights.sum() * num_samples

    targets = predicciones_perturbadas.view(-1, 1)

    # Bajamos un poco más el learning rate si ahora los gradientes son fuertes
    # O lo dejamos igual, pero vigilando.
    
    # ... (resto del código sigue igual)

    # --- CAMBIO CLAVE: MENOS REGULARIZACIÓN INICIAL ---
    l1_lambda = 1e-4  # Antes quizás era muy alto comparado con el gradiente

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- Nodos ---
        Eb = torch.matmul(E_stack, beta) 
        term_nodes = torch.matmul(alfa.t(), Eb).view(-1, 1)
        
        pred_approx = mu + (term_nodes * scale_nodes)

        # --- Edges ---
        if has_edges:
            Ad = torch.matmul(A_stack, delta)
            term_edges = torch.matmul(gamma.t(), Ad).view(-1, 1)
            pred_approx += (term_edges * scale_edges)

        squared_error = (targets - pred_approx)**2
        loss = (weights * squared_error).mean() # Usar mean ayuda a estabilizar respecto al batch size
        
        l1_reg = torch.norm(beta, 1) + torch.norm(alfa, 1) 
        if has_edges:
             l1_reg += torch.norm(delta, 1) + torch.norm(gamma, 1)
             
        total_loss = loss + (l1_lambda * l1_reg)

        total_loss.backward()
        optimizer.step()
        
        loss_val = total_loss.item()
        scheduler.step(loss_val)
        
        if verbose and epoch % 200 == 0:
             # Imprimimos también el bias para ver si se mueve
             logger.info(f"Epoch {epoch}: Loss {loss_val:.5f} | Mu: {mu.item():.3f}")

    return alfa.detach(), beta.detach(), (gamma.detach() if has_edges else None), (delta.detach() if has_edges else None), loss_val

def procesar_y_normalizar(tensor):
    # 1. Convertir Tensor a Numpy (Maneja si ya es None)
    if tensor is None:
        return None
    
    # .detach() para sacar del grafo, .cpu() para mover a RAM
    arr = tensor.detach().cpu().numpy().reshape(-1, 1)
    
    # 2. Valor Absoluto (Magnitud de la importancia)
    arr_abs = np.abs(arr)
    
    # 3. Escalar entre 0 y 1 (Min-Max Scaling)
    val_min = arr_abs.min()
    val_max = arr_abs.max()
    
    # Evitar división por cero si todos los valores son iguales (ej. todos 0)
    if val_max - val_min == 0:
        # Si max y min son iguales, devolvemos ceros (o unos si prefieres)
        return np.zeros_like(arr_abs)
    
    # Fórmula: (x - min) / (max - min)
    arr_norm = (arr_abs - val_min) / (val_max - val_min)
    
    return arr_norm

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
    cbar.ax.set_ylabel("Intensity", rotation=-90, va="bottom")

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

def get_feature_names(periodic_elements, hybridization_types):
    feature_names = []

    # 1️⃣ Tipos de átomo
    feature_names += [f"Atom_{el}" for el in periodic_elements]

    # 2️⃣ Grado
    feature_names.append("Degree_norm")

    # 3️⃣ Número de H
    feature_names.append("TotalHs_norm")

    # 4️⃣ Aromaticidad
    feature_names.append("IsAromatic")

    # 5️⃣ Hibridación
    feature_names += [f"Hybrid_{h}" for h in hybridization_types]

    return feature_names

def limpiar_columnas_zero(data, col_labels):
    # Detectar columnas que NO son todas ceros
    mask = ~(np.all(data == 0, axis=0))

    # Filtrar
    data_limpia = data[:, mask]
    col_labels_limpias = [label for label, keep in zip(col_labels, mask) if keep]

    return data_limpia, col_labels_limpias

def filtrar_features_presentes(alfa_np, feature_names, muestra_x):
    """
    Filtra la matriz de importancia (alfa) y los nombres de features
    para quedarse solo con aquellas que existen en la molécula original.
    """
    # 1. Asegurar que tenemos numpy array
    if hasattr(muestra_x, 'cpu'):
        x = muestra_x.cpu().numpy()
    else:
        x = muestra_x

    # 2. Crear máscara: True si la columna (feature) tiene algún valor != 0 en algún nodo
    # x tiene forma [Num_Nodos, Num_Features]
    mask = np.any(x != 0, axis=0)

    # 3. Verificación de seguridad de dimensiones
    if len(feature_names) != len(mask):
        logging.warning(f"Dimensión incorrecta: Nombres ({len(feature_names)}) vs Features ({len(mask)}). No se filtra.")
        return alfa_np, feature_names

    # 4. Filtrar nombres
    feature_names_filtered = [name for name, present in zip(feature_names, mask) if present]

    # 5. Filtrar matriz alfa
    alfa_np_filtered = alfa_np[mask]

    return alfa_np_filtered, feature_names_filtered

def plot_graph_with_beta(graph, beta, ax=None, cmap="YlOrRd", vmin=None, vmax=None, node_idx_map=None):
    """
    Dibuja el grafo molecular con los nodos coloreados según los valores de importancia β.

    Parámetros
    ----------
    graph : networkx.Graph
        Grafo molecular generado (por ejemplo, con parse_sdf()).
        Cada nodo debe tener atributos:
            - "pos": (x, y) para la posición 2D
            - "element": símbolo químico (e.g., "C", "O")
    beta : array-like
        Importancia de cada nodo (ordenada igual que los nodos en PyG Data).
    ax : matplotlib.axes.Axes, opcional
        Eje sobre el que dibujar. Si no se pasa, se crea uno nuevo.
    cmap : str, opcional
        Nombre del colormap de Matplotlib (por defecto "YlOrRd").
    vmin, vmax : float, opcional
        Límites inferior y superior de la escala de color.
    node_idx_map : dict, opcional
        Diccionario {nx_node_id: data_node_idx} para mapear correctamente los nodos
        de networkx a los índices de beta. Si no se pasa, se asume que el orden coincide.

    Devuelve
    --------
    ax : matplotlib.axes.Axes
        Eje con el grafo dibujado.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    beta = np.array(beta, dtype=float)

    if node_idx_map is not None:
        # Reordenar beta según el orden de nodos del grafo
        beta = np.array([beta[node_idx_map[n]] for n in graph.nodes])

    if vmin is None:
        vmin = float(np.min(beta))
    if vmax is None:
        vmax = float(np.max(beta))

    pos = {n: graph.nodes[n]["pos"] for n in graph.nodes}
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap(cmap)
    node_colors = [cmap(norm(b)) for b in beta]

    # Dibujar aristas
    nx.draw_networkx_edges(graph, pos, ax=ax, width=1.5, alpha=0.4, edge_color="gray")

    # Dibujar nodos coloreados por β
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=300,
        edgecolors="black",
        linewidths=0.6
    )

    # Etiquetas de elementos químicos
    labels = {n: graph.nodes[n].get("element", str(n)) for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels, font_size=9, ax=ax)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Importancia β", fontsize=10)

    ax.set_title("Grafo molecular coloreado por β")
    ax.axis("off")

    return ax

def plot_graph_with_importance(graph, node_importance, edge_importance=None, edge_index=None, ax=None, cmap="YlOrRd", node_idx_map=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # --- 1. PREPARACIÓN DE NODOS ---
    beta = np.array(node_importance, dtype=float)
    if node_idx_map is not None:
        # Reordenar beta para que coincida con los nodos del grafo NX
        beta_mapped = []
        for n in graph.nodes:
            # node_idx_map[n] da el índice en PyG
            idx = node_idx_map.get(n) if isinstance(n, str) else node_idx_map.get(str(n))
            # Si no encuentra (caso raro), usa 0
            val = beta[idx] if idx is not None and idx < len(beta) else 0.0
            beta_mapped.append(val)
        beta = np.array(beta_mapped)

    pos = {n: graph.nodes[n]["pos"] for n in graph.nodes}
    
    # Normalización Nodos
    vmin_n, vmax_n = beta.min(), beta.max()
    norm_n = mcolors.Normalize(vmin=vmin_n, vmax=vmax_n)
    cmap_obj = plt.cm.get_cmap(cmap)
    node_colors = [cmap_obj(norm_n(b)) for b in beta]

    # --- 2. PREPARACIÓN DE ARISTAS ---
    edge_colors = []
    edge_widths = []
    
    if edge_importance is not None and edge_index is not None:
        delta = np.array(edge_importance, dtype=float)
        
        # A) Crear mapa: (u_idx, v_idx) -> delta_value
        # edge_index debe ser tensor o array de shape [2, M]
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.cpu().numpy()
            
        imp_dict = {}
        for i in range(len(delta)):
            u = int(edge_index[0, i])
            v = int(edge_index[1, i])
            val = delta[i]
            # Guardamos ambas direcciones por seguridad
            imp_dict[(u, v)] = val
            imp_dict[(v, u)] = val

        # B) Normalización para aristas
        # Si todo es 0, evitamos división por cero
        if delta.max() > delta.min():
            norm_e = mcolors.Normalize(vmin=delta.min(), vmax=delta.max())
        else:
            norm_e = mcolors.Normalize(vmin=0, vmax=1)

        # C) Asignar color a cada arista de NetworkX
        nx_edges = list(graph.edges())
        
        for u, v in nx_edges:
            # Obtener índices reales de PyG
            # graph.nodes son strings '0', '1'... o ints 0, 1...
            u_real = node_idx_map.get(str(u)) if node_idx_map else int(u)
            v_real = node_idx_map.get(str(v)) if node_idx_map else int(v)

            # Buscar valor en el diccionario
            val = imp_dict.get((u_real, v_real), 0.0)
            
            # Obtener color
            rgba = cmap_obj(norm_e(val))
            edge_colors.append(rgba)
            
            # Opcional: Grosor basado en importancia
            # width = 1.0 + 4.0 * norm_e(val) 
            edge_widths.append(2.0) 

        # Dibujar aristas coloreadas
        nx.draw_networkx_edges(
            graph, pos, ax=ax, 
            edgelist=nx_edges,
            edge_color=edge_colors, 
            width=edge_widths,
            alpha=0.8
        )
        
        # Colorbar auxiliar para aristas (opcional, o usamos la misma)
        # sm_e = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm_e)
        # sm_e.set_array([])
        # plt.colorbar(sm_e, ax=ax, shrink=0.6, label="Edge Importance")

    else:
        # Fallback: Aristas grises si no hay datos
        nx.draw_networkx_edges(graph, pos, ax=ax, width=1.5, alpha=0.4, edge_color="gray")

    # --- 3. DIBUJAR RESTO ---
    # Dibujar Nodos (encima de las aristas)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=300, edgecolors="black")

    # Etiquetas
    labels = {n: graph.nodes[n].get("element", str(n)) for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels, font_size=9, ax=ax)
    
    # Colorbar principal
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm_n)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.8, label="Importance (Alpha/Beta)")
    
    ax.axis("off")
    return ax
