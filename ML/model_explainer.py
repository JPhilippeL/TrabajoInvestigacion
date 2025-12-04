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
from ML.data_processing import mol_to_graph_data, onehot_to_indices
from rdkit import Chem
from core.sdf_converter import parse_sdf
import matplotlib.gridspec as gridspec

SIGMADIST = 1
MININICIAL = sys.float_info.max
logger = logging.getLogger(__name__)

# Función para dada una muestra (x), genere una muestra perturbada (Z)
# Dado un vector binario de las características a perturbar
def perturb_features_sample(data, feature_mask=[1, 1, 1, 1, 1, 1], noise_level=0.05):
    data_new = data.clone()
    x = data_new.x
    num_nodes = x.shape[0]

    # Probabilidad de que un nodo/arista específico sea modificado.
    # Un 15% - 20% es razonable para mantener la estructura general.
    PERTURB_PROB = 0.15 
    
    # === DEFINICIÓN DE INDICES ===
    len_atom = len(periodic_elements)
    len_hybrid = len(hybridization_types)
    
    start_atom, end_atom = 0, len_atom
    start_hybrid = x.shape[1] - len_hybrid
    end_hybrid = x.shape[1]
    
    # Rellenar máscara
    if len(feature_mask) < 6:
        feature_mask = list(feature_mask) + [False] * (6 - len(feature_mask))

    # ==========================================
    # 1. PERTURBACIÓN DE NODOS
    # ==========================================
    for i in range(num_nodes):
        
        # A. TIPO DE ÁTOMO (One-Hot) - SOLO si toca la lotería (rand < PERTURB_PROB)
        if feature_mask[0] and torch.rand(1).item() < PERTURB_PROB:
            onehot = x[i, start_atom:end_atom]
            onehot[:] = 0
            new_idx = torch.randint(0, len_atom, (1,))
            onehot[new_idx] = 1

        # B. FEATURES CONTINUAS (Ruido)
        # El ruido continuo SÍ se puede aplicar a todos (es suave), 
        # o también puedes hacerlo probabilístico. Aquí lo dejo a todos pero suave.
        if feature_mask[1]:
            indices_continuous = [0, 1, 3, 4] 
            vals = x[i, end_atom:start_hybrid]
            noise = noise_level * torch.randn(len(indices_continuous))
            for k, idx_rel in enumerate(indices_continuous):
                vals[idx_rel] += noise[k]

        # C. FEATURES BINARIAS (Flip) - CRÍTICO: Hacerlo Probabilístico
        if feature_mask[2]:
            indices_binary = [2, 5, 6] # Aromatic, Donor, Acceptor
            vals = x[i, end_atom:start_hybrid]
            
            for idx_rel in indices_binary:
                # Solo invertimos el bit con probabilidad PERTURB_PROB
                if torch.rand(1).item() < PERTURB_PROB:
                    vals[idx_rel] = 1.0 - vals[idx_rel]

        # D. HIBRIDACIÓN (One-Hot) - Probabilístico
        if feature_mask[3] and torch.rand(1).item() < PERTURB_PROB:
            onehot = x[i, start_hybrid:end_hybrid]
            onehot[:] = 0
            new_idx = torch.randint(0, len_hybrid, (1,))
            onehot[new_idx] = 1

    data_new.x = x

    # ==========================================
    # 2. PERTURBACIÓN DE ARISTAS
    # ==========================================
    if data_new.edge_attr is not None:
        edge_attr = data_new.edge_attr
        edge_index = data_new.edge_index
        num_edges = edge_attr.shape[0]
        
        edge_map = {(edge_index[0, k].item(), edge_index[1, k].item()): k for k in range(num_edges)}
        dist_idx = -1 
        num_bond_cols = edge_attr.shape[1] - 1 

        for i in range(num_edges):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if u > v: continue 

            modified = False
            
            # Perturbar Tipo Enlace (Probabilístico)
            if feature_mask[4] and torch.rand(1).item() < PERTURB_PROB:
                onehot_bond = edge_attr[i, :num_bond_cols]
                onehot_bond[:] = 0
                new_bond_idx = torch.randint(0, num_bond_cols, (1,))
                onehot_bond[new_bond_idx] = 1
                modified = True
            
            # Perturbar Distancia (Siempre un poco de ruido está bien, o hazlo probabilístico)
            if feature_mask[5]: 
                noise = noise_level * torch.randn(1).item()
                edge_attr[i, dist_idx] += noise
                edge_attr[i, dist_idx] = torch.clamp(edge_attr[i, dist_idx], min=0.0)
                modified = True

            if modified and (v, u) in edge_map:
                sym_idx = edge_map[(v, u)]
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

def obtener_lime(checkpoint_path, sdf_path, feature_mask = [1, 1, 1, 1, 1, 1], num_samples=50, noise_level=0.05, device='cpu'):
    
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    muestra = mol_to_graph_data(mol, 'one_hot')

    # Mapear node index -> atom idx
    # node_to_atomidx = {i: atom.GetIdx() for i, atom in enumerate(mol.GetAtoms())}
    # Imprimir para verificación
    # for i in range(muestra.num_nodes):
    #    atom = mol.GetAtomWithIdx(node_to_atomidx[i])
    #    logger.info(f"Node {i} -> AtomIdx {atom.GetIdx()} ({atom.GetSymbol()})")
    
    # Generar muestras perturbadas
    perturbed_samples = generate_perturbed_samples(muestra, feature_mask, num_samples, noise_level)

    # Obtener modelo
    model, device, target_name = cargar_modelo(checkpoint_path)

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
    
    # ==========================================================================
    # PROCESAMIENTO DE MATRICES (Nueva Lógica)
    # ==========================================================================
    
    # 1. ALFA (Node Features) -> Filtrar -> Ordenar -> Normalizar
    node_feature_names = get_feature_names(periodic_elements, hybridization_types)
    alfa_sorted, row_labels_alfa = procesar_features_ordenadas(
        alfa, node_feature_names, muestra.x
    )
    col_labels_alfa = [""]

    # 2. GAMMA (Edge Features) -> Filtrar -> Ordenar -> Normalizar
    # Reemplaza a Beta en el segundo heatmap
    if muestra.edge_attr is not None:
        num_bond_types = muestra.edge_attr.shape[1] - 1 # El último es distancia
        edge_feature_names = ["Single", "Double", "Triple", "Aromatic"] + ["Distance"]
        
        gamma_sorted, row_labels_gamma = procesar_features_ordenadas(
            gamma, edge_feature_names, muestra.edge_attr
        )
    else:
        gamma_sorted = np.array([])
        row_labels_gamma = []
    col_labels_gamma = [""]

    # --- BETA (Nodos) ---
    beta_np = tensor_to_abs_numpy(beta)
    # CAMBIO: Usar normalizar_max para ser consistente con los heatmaps
    beta_np = normalizar_max(beta_np)  

    # --- DELTA (Aristas) ---
    if delta is not None:
        delta_np = tensor_to_abs_numpy(delta)
        # CAMBIO: Usar normalizar_max para evitar que una arista desaparezca si hay pocas
        delta_normalized = normalizar_max(delta_np) 
    else:
        delta_normalized = np.array([])
    
    # ==========================================================================
    # VISUALIZACIÓN
    # ==========================================================================
    HEIGHT_GRAPH = 10.0   
    HEIGHT_PER_ROW = 0.4
    
    num_rows_alfa = alfa_sorted.shape[0] if alfa_sorted is not None else 0
    num_rows_gamma = gamma_sorted.shape[0] if gamma_sorted is not None else 0
    max_rows = max(num_rows_alfa, num_rows_gamma)
    
    height_heatmaps = (max_rows * HEIGHT_PER_ROW) + 2.0 
    total_height = HEIGHT_GRAPH + height_heatmaps

    fig = plt.figure(figsize=(16, total_height))
    
    main_title = f"LIME Explanation for: **{mol_name}**\nModel Prediction: **{prediccion_original:.4f}** ({target_name})"
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[HEIGHT_GRAPH, height_heatmaps], hspace=0.2)

    # 1. GRAFO (Fila 0) - Usamos Delta Normalizado
    ax_graph = fig.add_subplot(gs[0, :])
    graph = parse_sdf(sdf_path)
    node_idx_map = {str(atom.GetIdx()): atom.GetIdx() for atom in mol.GetAtoms()}
    
    plot_graph_with_importance(
        graph, 
        node_importance=beta_np.flatten(), 
        edge_importance=delta_normalized.flatten(), # Delta normalizado (max=1)
        edge_index=muestra.edge_index,
        ax=ax_graph, 
        node_idx_map=node_idx_map,
        cmap="plasma"
    )

    # 2. HEATMAPS (Fila 1)
    
    # -- Heatmap ALFA (Node Features) --
    ax_alfa = fig.add_subplot(gs[1, 0])
    if alfa_sorted is not None and len(alfa_sorted) > 0:
        im_a, _ = heatmap(alfa_sorted, row_labels_alfa, col_labels_alfa, ax=ax_alfa, cmap="plasma", aspect='auto')
        annotate_heatmap(im_a, alfa_sorted, textcolors=("white", "black"))
        ax_alfa.set_title("Node Feature Importance (Alpha)", fontsize=14)
    else:
        ax_alfa.text(0.5, 0.5, "No significant node features", ha='center')

    # -- Heatmap GAMMA (Edge Features) -- Reemplaza a Beta
    ax_gamma = fig.add_subplot(gs[1, 1])
    if gamma_sorted is not None and len(gamma_sorted) > 0:
        im_g, _ = heatmap(gamma_sorted, row_labels_gamma, col_labels_gamma, ax=ax_gamma, cmap="plasma", aspect='auto')
        annotate_heatmap(im_g, gamma_sorted, textcolors=("white", "black"))
        ax_gamma.set_title("Edge Feature Importance (Gamma)", fontsize=14)
    else:
        ax_gamma.text(0.5, 0.5, "No edge features / Graph has no edges", ha='center')

    # Guardar
    model_name = checkpoint_path.split('/')[-1].split('.')[0]
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    model_results_dir = os.path.join(RESULTADOS_DIR, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    plotfilename = os.path.join(model_results_dir, f"{model_name}_for_{mol_name}_lime.png")

    plt.savefig(plotfilename, bbox_inches='tight')
    plt.close(fig)

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
    scale_nodes = 1.0 / d_nodes * N_nodes
    
    # Para Edges
    if has_edges:
        # Evitamos división por cero si N_edges es muy pequeño
        denom_edges = d_edges * M_edges
        scale_edges = 1.0 / denom_edges if denom_edges > 0 else 1.0
    else:
        scale_edges = 0.0

    # 3. Inicialización (Un poco más grande para ayudar al gradiente)
    alfa = nn.Parameter(torch.randn(d_nodes, 1, device=device) * 0.1)
    beta = nn.Parameter(torch.randn(N_nodes, 1, device=device) * 0.1)
    
    mean_pred = predicciones_perturbadas.mean().item()
    # mu = nn.Parameter(torch.tensor([mean_pred], device=device, dtype=torch.float))
    mu = nn.Parameter(torch.tensor([0.0], device=device, dtype=torch.float))

    # params = [alfa, beta, mu]
    params = [alfa, beta]
    
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

    # MEJORA 1: Usar AdamW en lugar de Adam estándar
    # weight_decay ayuda a mantener los pesos controlados sin ser tan agresivo como L1
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    
    # MEJORA 2: Scheduler tipo Coseno
    # Esto baja el LR suavemente desde 0.05 hasta 0 al final de las epochs.
    # Evita que se estanque tan pronto.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Variables para calcular R^2 al final
    initial_loss = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- Nodos ---
        Eb = torch.matmul(E_stack, beta) 
        term_nodes = torch.matmul(alfa.t(), Eb).view(-1, 1)
        
        pred_approx = (term_nodes * scale_nodes) # + mu
        # pred_approx = term_nodes * scale_nodes

        # --- Edges ---
        if has_edges:
            Ad = torch.matmul(A_stack, delta)
            term_edges = torch.matmul(gamma.t(), Ad).view(-1, 1)
            pred_approx += (term_edges * scale_edges)

        squared_error = (targets - pred_approx)**2
        loss = (weights * squared_error).mean() # Usar mean ayuda a estabilizar respecto al batch size
        
        # Guardamos el loss inicial real (sin regularización) para comparar
        if epoch == 0:
            initial_loss = loss.item()

        l1_reg = torch.norm(beta, 1) + torch.norm(alfa, 1) 
        if has_edges:
             l1_reg += torch.norm(delta, 1) + torch.norm(gamma, 1)
             
        total_loss = loss + (l1_lambda * l1_reg)

        total_loss.backward()
        optimizer.step()
        
        loss_val = total_loss.item()
        scheduler.step()
        
        if verbose and epoch % 200 == 0:
             # Imprimimos también el bias para ver si se mueve
             logger.info(f"Epoch {epoch}: Loss {loss_val:.5f} | Mu: {mu.item():.3f}")
    
    # --- MEJORA 3: CÁLCULO DE R^2 FINAL ---
    # R^2 = 1 - (Loss Final / Loss Inicial)
    # Esto te dice el % de la varianza explicada.
    r_squared = 1.0 - (loss_val / (initial_loss + 1e-8))
    logger.info(f"--- Entrenamiento Finalizado ---")
    logger.info(f"R² (Varianza Explicada): {r_squared:.2%} (Ideal > 80%)")

    return alfa.detach(), beta.detach(), (gamma.detach() if has_edges else None), (delta.detach() if has_edges else None), loss_val

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

    # 1️⃣ Tipos de átomo (One-Hot)
    # Coincide con: one_of_k_encoding_unk(atom.GetSymbol(), periodic_elements)
    feature_names += [f"Atom_{el}" for el in periodic_elements]

    # 2️⃣ Features Escalares (El "Sándwich" Central)
    # El orden AQUÍ debe ser idéntico al return de get_atom_features
    feature_names.append("Degree_norm")      # index relativo: 0
    feature_names.append("TotalHs_norm")     # index relativo: 1
    feature_names.append("IsAromatic")       # index relativo: 2
    
    # --- NUEVAS FEATURES ---
    feature_names.append("FormalCharge")     # index relativo: 3
    feature_names.append("GasteigerCharge")  # index relativo: 4
    feature_names.append("IsHDonor")         # index relativo: 5
    feature_names.append("IsHAcceptor")      # index relativo: 6

    # 3️⃣ Hibridación (One-Hot)
    # Coincide con: one_of_k_encoding_unk(atom.GetHybridization().name, hybridization_types)
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

def plot_graph_with_importance(graph, node_importance, edge_importance=None, edge_index=None, ax=None, cmap="YlOrRd", node_idx_map=None):
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
    
    # Normalización Nodos (Ahora con normalizar_max como acordamos)
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
            elif 'SINLE' in b_type_str:
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

    # --- 3. DIBUJAR NODOS Y ETIQUETAS (Igual que antes) ---
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=300, edgecolors="black")
    labels = {n: graph.nodes[n].get("element", str(n)) for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels, font_size=9, ax=ax)
    
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm_n)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.8, label="Importance")
    
    ax.axis("off")
    return ax

def tensor_to_abs_numpy(tensor):
    """Convierte tensor a numpy, toma valor absoluto."""
    if tensor is None: return None
    return np.abs(tensor.detach().cpu().numpy().reshape(-1, 1))

def normalizar_max(arr):
    """
    Normaliza dividiendo por el máximo absoluto.
    - El máximo será 1.0
    - El 0 real se queda en 0.
    - Mantiene la proporción real entre features.
    """
    if arr is None or len(arr) == 0: return arr
    
    # Usamos max() del valor absoluto, que ya viene calculado en 'arr'
    val_max = arr.max()
    
    if val_max == 0:
        return np.zeros_like(arr)
        
    return arr / val_max

def procesar_features_ordenadas(importance_tensor, feature_names, input_data):
    """
    Procesa features para Heatmaps usando Max Scaling.
    """
    if importance_tensor is None:
        return None, []
    
    # 1. Obtener magnitudes crudas (Valor Absoluto)
    raw_imp = tensor_to_abs_numpy(importance_tensor)
    
    # 2. Filtrar (Cortar features que no existen en la molécula)
    if hasattr(input_data, 'cpu'):
        x = input_data.cpu().numpy()
    else:
        x = input_data
        
    if x.shape[1] != raw_imp.shape[0]:
        mask = np.ones(raw_imp.shape[0], dtype=bool)
    else:
        mask = np.any(x != 0, axis=0)

    filtered_imp = raw_imp[mask]
    filtered_names = np.array(feature_names)[mask]
    
    if len(filtered_imp) == 0:
        return np.array([]), []

    # 3. Ordenar (Mayor a menor)
    sort_idx = np.argsort(filtered_imp.flatten())[::-1]
    
    sorted_imp = filtered_imp[sort_idx]
    sorted_names = filtered_names[sort_idx]
    
    # 4. NORMALIZAR CON MAX (El cambio clave)
    # Esto soluciona el problema de Gamma (2 features) y es más honesto para Alfa (11 features)
    final_imp = normalizar_max(sorted_imp)
    
    return final_imp, sorted_names.tolist()