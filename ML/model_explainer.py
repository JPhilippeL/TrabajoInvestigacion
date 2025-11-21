# model_explainer.py
from ML.model_tester import cargar_modelo, predecir_molecula
import torch
import matplotlib.pyplot as plt
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
            x[i, start_degree:end_hs] += noise_level * torch.randn_like(x[i, start_degree:end_hs])
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
            if feature_mask[5]:
                edge_attr[i, idx_dist] += noise_level * torch.randn_like(edge_attr[i, idx_dist])
                modified = True

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
    for i in range(muestra.num_nodes):
        atom = mol.GetAtomWithIdx(node_to_atomidx[i])
        logger.info(f"Node {i} -> AtomIdx {atom.GetIdx()} ({atom.GetSymbol()})")
    
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
    alfa, beta, gamma, delta, loss = obtener_argmin(feature_distances, predicciones_perturbadas, E_t_list, A_t_list, 0.01, 500, True)
    
    # --- Convertir tensores a numpy ---
    alfa_np = alfa.detach().cpu().numpy().reshape(-1, 1)  # d x 1  → columna
    beta_np = beta.detach().cpu().numpy().reshape(-1, 1)  # N x 1  → columna

    # Nuevos tensores
    gamma_np = gamma.cpu().numpy().reshape(-1, 1) # Importancia de features de edge (e.g., Doble vs Simple)
    delta_np = delta.cpu().numpy().reshape(-1, 1) # Importancia de cada enlace específico

    # --- Etiquetas ---
    feature_names = get_feature_names(periodic_elements, hybridization_types)
    row_labels_alfa = feature_names
    col_labels_alfa = [""]

    row_labels_beta = [f"Node {i}" for i in range(len(beta_np))]
    col_labels_beta = [""]

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # --- Nuevo: Añadir título principal a la figura ---
    # Crear el título principal de la figura
    main_title = f"LIME Explanation for: **{mol_name}**\nModel Prediction: **{prediccion_original:.4f}** ({target_name})"
    fig.suptitle(main_title, fontsize=14, fontweight='bold')

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])  # ocupa toda la fila inferior

    # α y β arriba
    im_a, _ = heatmap(alfa_np, row_labels_alfa, col_labels_alfa, ax=ax1, cmap="plasma")
    annotate_heatmap(im_a, alfa_np, textcolors=("white", "black"))
    im_b, _ = heatmap(beta_np, row_labels_beta, col_labels_beta, ax=ax2, cmap="plasma")
    annotate_heatmap(im_b, beta_np, textcolors=("white", "black"))

    # Grafo abajo centrado y grande
    graph = parse_sdf(sdf_path)
    node_idx_map = {str(atom.GetIdx()): atom.GetIdx() for atom in mol.GetAtoms()}
    #plot_graph_with_beta(graph, beta_np.flatten(), ax=ax3, cmap="YlOrRd", node_idx_map=node_idx_map)
    
    # Llamada actualizada al plotter
    plot_graph_with_importance(
        graph, 
        node_importance=beta_np.flatten(), 
        edge_importance=delta_np.flatten(), 
        edge_index=muestra.edge_index,  # <--- IMPORTANTE: Necesario para mapear delta
        ax=ax3, 
        node_idx_map=node_idx_map,
        cmap="plasma"  # <--- CAMBIO AQUÍ (o "plasma", "cividis")
    )

    plt.tight_layout()



    # Obtener el nombre del checkpoint
    model_name = checkpoint_path.split('/')[-1]
    # quitarle la extension
    model_name = model_name.split('.')[0]

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    model_results_dir = os.path.join(RESULTADOS_DIR, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    plotfilename = os.path.join(model_results_dir, f"{model_name}_for_{mol_name}_lime.png")

    # Guardar la figura en el resultados dir de este checkpoint
    plt.tight_layout()
    plt.savefig(plotfilename)
    plt.close(fig)

    # Return de la imagen guardada
    return plotfilename

def obtener_argmin(feature_distances, predicciones_perturbadas, 
                   E_t_list, A_t_list, constNodos = 1, constEdges = 1,
                   lr=0.05, 
                   epochs=2000, 
                   verbose=True):
    
    device = predicciones_perturbadas.device  # Alias para escribir menos
    num_samples = len(E_t_list)
    d_nodes = E_t_list[0].shape[0]
    N_nodes = E_t_list[0].shape[1]
    
    # Manejo si no hay aristas
    if len(A_t_list) > 0:
        d_edges = A_t_list[0].shape[0]
        M_edges = A_t_list[0].shape[1]
        has_edges = True
    else:
        d_edges = 1
        M_edges = 1
        has_edges = False

    feature_distances = torch.tensor(feature_distances, dtype=torch.float, device=device).unsqueeze(1)

    # --- CORRECCIÓN DE INICIALIZACIÓN ---
    # Creamos los datos aleatorios, los movemos al device, *detach()* para olvidar la operación,
    # y finalmente activamos el gradiente.

    # 1. ALFA
    alfa_init = torch.randn(d_nodes, device=device, dtype=torch.float) * 0.1
    alfa = alfa_init.detach().requires_grad_(True)

    # 2. BETA
    beta_init = torch.randn(N_nodes, device=device, dtype=torch.float) * 0.1
    beta = beta_init.detach().requires_grad_(True)
    
    # 3. BIAS (MU)
    mean_pred = predicciones_perturbadas.mean().item()
    # Crear tensor directamente con el valor float (es seguro)
    mu = torch.tensor([mean_pred], device=device, dtype=torch.float, requires_grad=True)

    if has_edges:
        # 4. GAMMA
        gamma_init = torch.randn(d_edges, device=device, dtype=torch.float) * 0.1
        gamma = gamma_init.detach().requires_grad_(True)
        
        # 5. DELTA
        delta_init = torch.randn(M_edges, device=device, dtype=torch.float) * 0.1
        delta = delta_init.detach().requires_grad_(True)
        
        params = [alfa, beta, gamma, delta, mu]
    else:
        # Dummies que no requieren gradiente para evitar errores si no se usan
        gamma = torch.tensor(0.0, device=device)
        delta = torch.tensor(0.0, device=device)
        params = [alfa, beta, mu]

    # Optimizador
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0) 
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=False)

    exp_weights = torch.exp(-feature_distances**2)
    
    best_loss = float('inf')
    patience_counter = 0
    early_stopping_limit = 200 

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0

        for i in range(num_samples):
            # Término Nodos
            term_nodes = torch.matmul(alfa, torch.matmul(E_t_list[i], beta))
            
            approx_pred = mu +  constNodos * term_nodes
            
            # Término Edges
            if has_edges:
                term_edges = torch.matmul(gamma, torch.matmul(A_t_list[i], delta))
                approx_pred += constEdges * term_edges
            
            pred_z = predicciones_perturbadas[i]
            w = exp_weights[i]
            
            total_loss += w * (pred_z - approx_pred)**2

        # Regularización L1 Suave
        l1_lambda = 0.001 
        l1_norm = torch.norm(beta, 1) + (torch.norm(delta, 1) if has_edges else 0)
        
        final_loss = total_loss + l1_lambda * l1_norm

        final_loss.backward()
        optimizer.step()
        
        current_loss_val = total_loss.item()
        scheduler.step(current_loss_val)

        # Early Stopping
        if current_loss_val < best_loss - 1e-4:
            best_loss = current_loss_val
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_limit:
            if verbose:
                logger.info(f"Converged early at epoch {epoch}. Loss: {best_loss:.4f}")
            break

        if verbose and (epoch % 100 == 0 or epoch == epochs-1):
            logger.info(f"Epoch {epoch}/{epochs} - Loss: {current_loss_val:.4f} - Bias: {mu.item():.2f}")

    return alfa.detach(), beta.detach(), gamma.detach(), delta.detach(), best_loss



def heatmap(data, row_labels, col_labels, ax, **kwargs):
    if ax is None:
        ax = plt.gca()
    cbar_kw = {}
    
    # Crear heatmap
    cbar_kw = {}
    im = ax.imshow(data, aspect=0.2, **kwargs)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel("Intensity", rotation=-90, va="bottom")

    # Show all ticks and label them
    ax.set_xticks(range(len(col_labels)), 
                  labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

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

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

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
