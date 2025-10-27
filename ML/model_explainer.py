# model_explainer.py
from ui.utils import periodic_elements, hybridization_types
from ML.model_tester import cargar_modelo, predecir_molecula
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
from ui.utils import RESULTADOS_DIR
import os
import sys

SIGMADIST = 1
MININICIAL = sys.float_info.max

# Función para dada una muestra (x), genere una muestra perturbada (Z)
# Dado un vector binario de las características a perturbar
# Vector mascara: [perturbar_tipo_atomo, perturbar_grado, perturbar_aromaticidad, perturbar_hibridacion]
# TO DO: VER LO DE  DISTRIBUCION GAUSSIANA DISCRETA
def perturb_features_sample(data, feature_mask, noise_level=0.05):
    data_new = data.clone()
    x = data_new.x
    # Definir slices de features según tu vector
    start_atom, end_atom = 0, len(periodic_elements)
    start_degree, end_degree = end_atom, end_atom+1
    start_hs, end_hs = end_degree, end_degree+1
    aromatic_idx = end_hs
    start_hybrid, end_hybrid = aromatic_idx+1, aromatic_idx+1+len(hybridization_types)
    auxiliar = x.shape[0]

    for i in range(auxiliar):
        if feature_mask[0]:  # Perturbar tipo de átomo
            onehot = x[i, start_atom:end_atom]
            onehot[:] = 0
            new_idx = torch.randint(0, len(periodic_elements), (1,))
            onehot[new_idx] = 1
        if feature_mask[1]:  # Perturbar grado
            x[i, start_degree:end_hs] += noise_level * torch.randn_like(x[i, start_degree:end_hs])
        if feature_mask[2]:  # Perturbar aromaticidad
            x[i, aromatic_idx] = 1 - x[i, aromatic_idx]  # flip
        if feature_mask[3]:  # Perturbar hibridación
            onehot = x[i, start_hybrid:end_hybrid]
            onehot[:] = 0
            new_idx = torch.randint(0, len(hybridization_types), (1,))
            onehot[new_idx] = 1

    data_new.x = x
    return data_new

# Función para generar múltiples muestras perturbadas
# La distribución de las muestras aleatorias tienen que seguir una distribucion gaussiana
def generate_perturbed_samples(data, feature_mask, num_samples=50, noise_level=0.05):
    perturbed_samples = []
    for _ in range(num_samples):
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

def compute_alfa_beta_alfa_ET_beta(data):
    
    matrizE = data.x.clone()  # [N, d]

    # --- α: suma por columnas (features) ---
    alfa = matrizE.sum(dim=0)  # [d]

    # --- β: suma por filas (nodos) ---
    beta = matrizE.sum(dim=1, keepdim=True)  # [N,1]

    # --- α * matrizE^T * β ---
    # matrizE^T: [d, N], beta: [N,1] => matmul => [d,1]
    matrizE_t = matrizE.t()  # [d, N]
    res = torch.matmul(matrizE_t, beta)  # [d,1]

    alfa_ET_beta_scalar = torch.matmul(alfa.unsqueeze(0), res)  # [1, d] × [d, 1] → [1, 1]

    # beta * alfa
    beta_alfa = beta @ alfa.unsqueeze(0)  # [N,1] @ [1,d] => [N,d]



    return {
        'matrizE': matrizE,                     # [N, d]
        'alfa': alfa,               # [d]
        'beta': beta,               # [N,1]
        'alfa_ET_beta_scalar': alfa_ET_beta_scalar,  # [1,1]
        'beta_alfa': beta_alfa      # [N,d]
    }

def obtener_lime(checkpoint_path, data_sample, feature_mask, num_samples=50, noise_level=0.05, device='cpu'):
    # Generar muestras perturbadas
    perturbed_samples = generate_perturbed_samples(data_sample, feature_mask, num_samples, noise_level)

    # Obtener modelo
    model, device, target_name = cargar_modelo(checkpoint_path)

    # Predecir las muestras perturbadas
    predicciones_perturbadas = []
    for perturbed in perturbed_samples:
        perturbed = perturbed.to(device)
        pred = predecir_molecula(model, perturbed, device)
        predicciones_perturbadas.append(pred)

    # Convertir a tensor [num_samples,1]
    predicciones_perturbadas = torch.tensor(predicciones_perturbadas, dtype=torch.float, device=device).unsqueeze(1)

    # Calcular distancias entre la muestra original y las perturbaciones
    feature_distances = graph_feature_distance_list(data_sample, perturbed_samples, metric='euclidean')
    #feature_distances = embedding_distance_list(model, data_sample, perturbed_samples, device=device)

    anterior = MININICIAL
    # Wg = alfa, Z' = matrizE_t * beta
    # Hacer lista de todas las Wg y Z' para cada muestra perturbada
    for i, perturbed in enumerate(perturbed_samples):
        info_perturbada = compute_alfa_beta_alfa_ET_beta(perturbed)
        
        # Calcular distancia
        distancia = feature_distances[i]

        # Distancia = e^(-d(x, z)²)/SIGMADIST
        distancia_tensor = torch.tensor(distancia, dtype=torch.float, device=device)
        peso_distancia = torch.exp(- (distancia_tensor ** 2) / SIGMADIST)

        # Prediccion de z
        pred_z = predicciones_perturbadas[i]  # [1,1]
        # Wg * z' = alfa * (matrizE_t * beta)
        Wg_z_prime = info_perturbada['alfa_ET_beta_scalar'] # [1,1]

        # Restarlo y elevar al cuadrado
        # Pc(z) - Wg * z'
        diff = pred_z - Wg_z_prime  # [1,1]
        diff_squared = diff ** 2  # [1,1]

        # Multiplicar
        termino = peso_distancia * diff_squared  # [1,1]

        if termino < anterior:
            anterior = termino
            # quedarnos con el indice
            info_mejor = info_perturbada

    beta_alfa = info_mejor['beta_alfa']

    # Convertir a numpy para matplotlib
    beta_alfa = info_mejor['beta_alfa'].detach().cpu().numpy()  # [N,d]

    n, d = beta_alfa.shape
    row_labels = [f'Node {i}' for i in range(n)]
    # Obtener nombres de características
    feature_names = get_feature_names(periodic_elements, hybridization_types)
    

    # --- CREAR IMAGEN ---
    # Llamar a lo de limpiar columnas
    data_cleaned, col_labels_clean = limpiar_columnas_zero(beta_alfa, feature_names)

    fig, ax = plt.subplots(figsize=(20, 20))
    
    im, cbar = heatmap(data_cleaned, row_labels, col_labels_clean, ax=ax, cmap="YlGn")

    annotate_heatmap(im)

    # Obtener el nombre del checkpoint
    model_name = checkpoint_path.split('/')[-1]
    # quitarle la extension
    model_name = model_name.split('.')[0]

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    model_results_dir = os.path.join(RESULTADOS_DIR, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    plotfilename = os.path.join(model_results_dir, f"{model_name}_lime.png")

    # Guardar la figura en el resultados dir de este checkpoint
    plt.savefig(plotfilename)
    plt.close(fig)

    # Return de la imagen guardada
    return plotfilename


def heatmap(data, row_labels, col_labels, ax, **kwargs):
    if ax is None:
        ax = plt.gca()
    cbar_kw = {}
    
    # Crear heatmap
    cbar_kw = {}
    im = ax.imshow(data, aspect='auto', **kwargs)

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