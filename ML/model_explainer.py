# model_explainer.py
from ui.utils import periodic_elements, hybridization_types
from ML.model_tester import cargar_modelo, predecir_molecula
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

    alfa = info_mejor['alfa']
    beta = info_mejor['beta']
    beta_alfa = info_mejor['beta_alfa']

    # Convertir a numpy para matplotlib
    alfa = info_mejor['alfa'].detach().cpu().numpy()          # [d]
    beta = info_mejor['beta'].detach().cpu().numpy()          # [N,1]
    beta_alfa = info_mejor['beta_alfa'].detach().cpu().numpy()  # [N,d]


    # --- CREAR IMAGEN ---
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 8, 0.5], width_ratios=[8, 1], 
                        hspace=0.05, wspace=0.05)

    # Subplot para alfa (arriba, horizontal)
    ax_alfa = fig.add_subplot(gs[0, 0])
    im_alfa = ax_alfa.imshow(alfa.reshape(1, -1), cmap='RdYlGn', aspect='auto')
    ax_alfa.set_title('(Feature weights)', fontsize=12, pad=10)
    ax_alfa.set_yticks([])
    ax_alfa.set_xlabel('Features')
    # Añadir valores en las celdas
    for i in range(len(alfa)):
        ax_alfa.text(i, 0, f'{alfa[i]:.2f}', ha='center', va='center', fontsize=8)

    # Subplot para beta*alfa (centro, matriz principal)
    ax_main = fig.add_subplot(gs[1, 0])
    im_main = ax_main.imshow(beta_alfa, cmap='RdYlGn', aspect='auto')
    ax_main.set_title('β * Node-Feature importance matrix)', fontsize=12, pad=10)
    ax_main.set_ylabel('Nodes')
    ax_main.set_xlabel('Features')
    # Añadir valores en celdas (solo si no son demasiadas)
    if beta_alfa.shape[0] * beta_alfa.shape[1] < 500:
        for i in range(beta_alfa.shape[0]):
            for j in range(beta_alfa.shape[1]):
                ax_main.text(j, i, f'{beta_alfa[i,j]:.2f}', 
                            ha='center', va='center', fontsize=6)

    # Subplot para beta (derecha, vertical)
    ax_beta = fig.add_subplot(gs[1, 1])
    im_beta = ax_beta.imshow(beta, cmap='RdYlGn', aspect='auto')
    ax_beta.set_title('β', fontsize=12, pad=10, rotation=0)
    ax_beta.set_xticks([])
    ax_beta.set_ylabel('Nodes')
    # Añadir valores en las celdas
    for i in range(beta.shape[0]):
        ax_beta.text(0, i, f'{beta[i,0]:.2f}', ha='center', va='center', fontsize=8)

    # Colorbars
    cbar_alfa = plt.colorbar(im_alfa, ax=ax_alfa, orientation='horizontal', 
                            pad=0.1, fraction=0.05)
    cbar_main = plt.colorbar(im_main, ax=ax_main, orientation='vertical', 
                            pad=0.02, fraction=0.046)
    cbar_beta = plt.colorbar(im_beta, ax=ax_beta, orientation='vertical', 
                            pad=0.02, fraction=0.046)

    plt.suptitle('LIME Explanation for GNN', fontsize=14, y=0.98)
    # --- FIN CREAR IMAGEN ---

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

