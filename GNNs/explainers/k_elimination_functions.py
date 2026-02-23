import torch
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from ui.utils.constants import (
    EMBEDDING_INDICES, 
    UNKNOWN_ATOM_IDX, 
    UNKNOWN_HYBRID_IDX,
    EDGE_EMBEDDING_INDICES,
    UNKNOWN_BOND_IDX,
    periodic_elements,
    hybridization_types
)

# --- ALFA INDICES ---
def ocultar_features_nodos_indices(data, indices_features_a_ocultar):
    """
    MODO ALFA (INDICES): Perturba las features indicadas.
    - Categóricas (Átomo/Hibridación): Se fuerzan al índice 'Unknown'.
    - Continuas (Carga/Grado, etc): Se reemplazan por la media (o 0).
    """
    # 1. Clonamos x para no modificar el original
    x_mod = data.x.clone()
    
    # 2. Pre-calculamos la media por columna (solo para las continuas)
    feature_means = x_mod.mean(dim=0) 
    
    # 3. Iteramos sobre los índices que queremos ocultar
    # Es necesario iterar porque la lógica cambia según la columna
    if len(indices_features_a_ocultar) > 0:
        
        # Aseguramos que sea iterable simple (lista o array)
        if torch.is_tensor(indices_features_a_ocultar):
            lista_indices = indices_features_a_ocultar.cpu().numpy().tolist()
        else:
            lista_indices = indices_features_a_ocultar

        for feat_idx in lista_indices:
            feat_idx = int(feat_idx) # Seguridad
            
            # --- CASO A: TIPO DE ÁTOMO ---
            if feat_idx == EMBEDDING_INDICES["ATOM_SYMBOL"]:
                # Asignar el índice de Desconocido a todos los nodos
                x_mod[:, feat_idx] = UNKNOWN_ATOM_IDX
                
            # --- CASO B: HIBRIDACIÓN ---
            elif feat_idx == EMBEDDING_INDICES["HYBRIDIZATION"]:
                x_mod[:, feat_idx] = UNKNOWN_HYBRID_IDX
                
            # --- CASO C: CONTINUAS (El resto) ---
            else:
                # Opción 1: Usar la Media (Suavizado) -> Mantiene distribución
                x_mod[:, feat_idx] = feature_means[feat_idx]
                
                # Opción 2: Usar Cero -> Elimina la señal (Descomentar si prefieres)
                # x_mod[:, feat_idx] = 0.0

    # 4. Retornamos nuevo objeto Data
    new_data = data.clone()
    new_data.x = x_mod
    
    return new_data

# --- ALFA ONEHOT ---
def ocultar_features_nodos_onehot(data, indices_cols_a_ocultar):
    """
    MODO ALFA (ONE-HOT MIXTO):
    - Features One-Hot (Átomos/Hibridación): Se ponen a 0 (Zero Masking -> Unknown).
    - Features Continuas (Carga/Grado): Se reemplazan por la MEDIA de la columna.
    """
    x_mod = data.x.clone()
    
    # 1. Definir los límites de las secciones
    # Estructura asumen: [ ÁTOMOS (OneHot) | CONTINUAS | HIBRIDACIÓN (OneHot) ]
    
    n_atoms = len(periodic_elements)
    n_total = x_mod.shape[1]
    n_hybrid = len(hybridization_types)
    
    # Índices donde empieza la hibridación
    start_hybrid = n_total - n_hybrid 
    
    # 2. Pre-calcular las medias (solo se usarán para las continuas)
    feature_means = x_mod.mean(dim=0)
    
    if len(indices_cols_a_ocultar) > 0:
        
        # Normalizar a lista para iterar y comprobar rangos
        if torch.is_tensor(indices_cols_a_ocultar):
            indices = indices_cols_a_ocultar.cpu().numpy().tolist()
        else:
            indices = indices_cols_a_ocultar
            
        for idx in indices:
            idx = int(idx)
            
            # --- CASO A: ES ÁTOMO (One-Hot) ---
            if idx < n_atoms:
                x_mod[:, idx] = 0.0
                
            # --- CASO B: ES HIBRIDACIÓN (One-Hot) ---
            elif idx >= start_hybrid:
                x_mod[:, idx] = 0.0
                
            # --- CASO C: ES CONTINUA (El sándwich del medio) ---
            else:
                x_mod[:, idx] = feature_means[idx]
                
    new_data = data.clone()
    new_data.x = x_mod
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
def ocultar_features_aristas_indices(data, indices_features_a_ocultar):
    """
    MODO GAMMA (INDICES): Perturba las features de las aristas.
    - Categóricas (Tipo Enlace): Se fuerzan al índice 'Unknown/Other'.
    - Continuas (Distancia): Se reemplazan por la media.
    """
    if data.edge_attr is None:
        return data

    edge_attr_mod = data.edge_attr.clone()
    
    # Pre-calculamos la media para las features continuas (Distancia)
    feature_means = edge_attr_mod.mean(dim=0)
    
    if len(indices_features_a_ocultar) > 0:
        
        # Convertimos a lista simple para iterar
        if torch.is_tensor(indices_features_a_ocultar):
            lista_indices = indices_features_a_ocultar.cpu().numpy().tolist()
        else:
            lista_indices = indices_features_a_ocultar

        for feat_idx in lista_indices:
            feat_idx = int(feat_idx)
            
            # --- CASO A: TIPO DE ENLACE ---
            if feat_idx == EDGE_EMBEDDING_INDICES["BOND_TYPE"]:
                # Asignamos la categoría 'OTHER' / 'UNKNOWN'
                # Esto le dice al modelo: "Aquí hay una arista, pero no sé qué tipo es"
                edge_attr_mod[:, feat_idx] = UNKNOWN_BOND_IDX
                
            # --- CASO B: DISTANCIA (O cualquier otra continua) ---
            else:
                # Usamos la media para suavizar la distancia
                edge_attr_mod[:, feat_idx] = feature_means[feat_idx]
                
    return Data(x=data.x, edge_index=data.edge_index, 
                edge_attr=edge_attr_mod, batch=data.batch)


def ocultar_features_aristas_onehot(data, indices_cols_a_ocultar):
    """
    MODO GAMMA (ONE-HOT MIXTO):
    - Features One-Hot (Tipo Enlace): Se ponen a 0 (Zero Masking -> Unknown).
    - Features Continuas (Distancia): Se reemplazan por la MEDIA de la columna.
    """
    if data.edge_attr is None:
        return data

    edge_attr_mod = data.edge_attr.clone()
    
    # Identificamos el índice de la distancia
    # En tu código es siempre la última columna (-1)
    num_features = edge_attr_mod.shape[1]
    dist_idx = num_features - 1 
    
    # Pre-calcular medias
    feature_means = edge_attr_mod.mean(dim=0)
    
    if len(indices_cols_a_ocultar) > 0:
        
        # Convertir a lista simple para iterar (más seguro para lógica condicional)
        if torch.is_tensor(indices_cols_a_ocultar):
            indices = indices_cols_a_ocultar.cpu().numpy().tolist()
        else:
            indices = indices_cols_a_ocultar
            
        for idx in indices:
            idx = int(idx)
            
            # Validación de seguridad
            if idx >= num_features:
                continue

            # --- LÓGICA DE PERTURBACIÓN ---
            
            # CASO A: Es la Distancia (Continua)
            if idx == dist_idx:
                edge_attr_mod[:, idx] = feature_means[idx]
                
            # CASO B: Es Tipo de Enlace (One-Hot)
            else:
                edge_attr_mod[:, idx] = 0.0
        
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
                