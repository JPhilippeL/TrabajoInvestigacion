# model_explainer.py
from ui.utils import periodic_elements, hybridization_types
from ML.model_tester import cargar_modelo, predecir_molecula
import torch

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
    """
    Calcula la distancia promedio entre una muestra original x y cada 
    muestra perturbada de z_list usando las features de nodos.

    Args:
        x (torch_geometric.data.Data): muestra original.
        z_list (list[Data]): lista de muestras perturbadas.
        metric (str): 'euclidean' o 'cosine'

    Returns:
        list[float]: lista de distancias de x a cada z en z_list
    """
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

def compute_alfa_beta_alpha_ET_beta(data, periodic_elements, hybridization_types):
    """
    data: PyG Data object
    periodic_elements: lista de elementos químicos para one-hot
    hybridization_types: lista de tipos de hibridación para one-hot
    """

    matrizE = data.x.clone()  # [N, d]

    # --- α: suma por columnas (features) ---
    alfa = matrizE.sum(dim=0)  # [d]

    # --- β: suma por filas (nodos) ---
    beta = matrizE.sum(dim=1, keepdim=True)  # [N,1]

    # --- α * matrizE^T * β ---
    # matrizE^T: [d, N], beta: [N,1] => matmul => [d,1]
    matrizE_t = matrizE.t()  # [d, N]
    res = torch.matmul(matrizE_t, beta)  # [d,1]

    # α: [d] -> [d,1] para multiplicación elemento a elemento
    alfa_exp = alfa.unsqueeze(1)  # [d,1]
    alpha_ET_beta = alfa_exp * res  # [d,1]

    return {
        'matrizE': matrizE,                     # [N, d]
        'alfa': alfa,               # [d]
        'beta': beta,               # [N,1]
        'alpha_ET_beta': alpha_ET_beta  # [d,1]
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

    # 5️⃣ Calcular distancias entre la muestra original y las perturbaciones
    feature_distances = graph_feature_distance_list(data_sample, perturbed_samples, metric='euclidean')
    pesos_perturbaciones = torch.exp(torch.tensor(feature_distances, dtype=torch.float, device=device)).unsqueeze(1)

    # 6️⃣ Calcular z' para cada perturbación: alfa * E^T * beta
    Z_prime_list = []
    for perturbed in perturbed_samples:
        result = compute_alfa_beta_alpha_ET_beta(perturbed, periodic_elements, hybridization_types)
        z_prime = result['alpha_ET_beta']  # [num_features,1]
        Z_prime_list.append(z_prime.t())   # convertir a [1, num_features]

    # 7️⃣ Construir matriz de features de perturbaciones [num_samples, num_features]
    feature_matrix_perturbadas = torch.cat(Z_prime_list, dim=0)

    # 8️⃣ Crear matriz diagonal de pesos para regresión ponderada
    matriz_pesos = torch.diag(pesos_perturbaciones.squeeze())

    # 9️⃣ Resolver regresión lineal ponderada para obtener Wg
    perturbadasXpesos = feature_matrix_perturbadas.t() @ matriz_pesos
    Wg = torch.linalg.pinv(perturbadasXpesos @ feature_matrix_perturbadas) @ (perturbadasXpesos @ predicciones_perturbadas)

    return Wg, feature_matrix_perturbadas, predicciones_perturbadas, pesos_perturbaciones


