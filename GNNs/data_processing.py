#data_processing.py
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from ui.utils.constants import (
    BOND_TYPE_TO_INT, UNKNOWN_BOND_IDX, UNKNOWN_ATOM_IDX, UNKNOWN_HYBRID_IDX,
    periodic_elements, hybridization_types,
    ATOM_TYPE_TO_IDX, HYBRID_TO_IDX,
    RESULTADOS_DIR, N_BOND_TYPES, OTHER_EDGE_FEATURES,OTHER_NODE_FEATURES
)
from sklearn.model_selection import train_test_split
import math
import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from rdkit import Chem

# --- Definiciones de Patrones SMARTS para H-Bonds ---
# Donador: Generalmente N u O con al menos un H unido
HBD_PATTERN = Chem.MolFromSmarts('[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]')

# Aceptor: N u O con pares libres disponibles (definición estándar de Lipinski)
HBA_PATTERN = Chem.MolFromSmarts('[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]')

# Patrón para enlace rotable: Enlace simple (-), no en anillo (!@), entre átomos no terminales (!D1)
FLEXIBILITY_BOND_PATTERN = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')

import torch
from torch_geometric.data import Data
from rdkit import Chem
from typing import Any, List

# =====================================================================
# 1. DICCIONARIOS Y FUNCIONES ORIGINALES (Copiados tal cual)
# =====================================================================

ATOM_FEATURES = {
    'atomic_num': [1, 6, 7, 8, 9, 15, 16, 17, 35, 53],
    'formal_charge': [-1, 0, 1],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3
    ],
    'aromatic': [0, 1],
    'ring_size': [0, 3, 4, 5, 6, 7, 8],
    'h_bonding': [0, 1, 2]
}

BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'is_conjugated': [0, 1],
    'is_in_ring': [0, 1]
}

def one_hot_encoding(value: Any, feature_list: List) -> List[int]:
    if value not in feature_list:
        encoding = [0] * len(feature_list)
    else:
        encoding = [0] * len(feature_list)
        encoding[feature_list.index(value)] = 1
    return encoding

def get_atom_features(atom: Chem.Atom) -> List[int]:
    features = []
    features += one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    features += one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    features += [int(atom.GetIsAromatic())]
    
    ring_size = 0
    mol = atom.GetOwningMol()
    for ring in mol.GetRingInfo().AtomRings():
        if atom.GetIdx() in ring:
            ring_size = len(ring)
            break
    features += one_hot_encoding(ring_size, ATOM_FEATURES['ring_size'])
    
    h_bonding = 0
    if atom.GetAtomicNum() in [7, 8]:
        h_bonding = 1
    if atom.GetTotalNumHs() > 0 and atom.GetAtomicNum() in [7, 8]:
        h_bonding = 2
    features += one_hot_encoding(h_bonding, ATOM_FEATURES['h_bonding'])
    
    return features

def get_bond_features(bond: Chem.Bond) -> List[int]:
    features = []
    features += one_hot_encoding(bond.GetBondType(), BOND_FEATURES['bond_type'])
    features += [int(bond.GetIsConjugated())]
    features += [int(bond.IsInRing())]
    return features


# =====================================================================
# 2. TU FUNCIÓN DE GRAFO ACTUALIZADA
# =====================================================================

def mol_to_graph_data(mol: Chem.Mol) -> Data:
    """
    Convierte una molécula RDKit en un objeto Data de PyTorch Geometric
    usando estrictamente las nuevas features one-hot.
    """
    
    # === 1. NODOS (Átomos) ===
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    # PyTorch Geometric espera que las features de nodos sean floats
    x = torch.tensor(atom_features, dtype=torch.float)

    # === 2. ARISTAS (Enlaces) ===
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Obtenemos las features del enlace
        bond_feat = get_bond_features(bond)

        # Grafo no dirigido: agregamos (i, j) y (j, i)
        edge_index.append([i, j])
        edge_index.append([j, i])
        
        # Duplicamos las features para cada dirección
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)

    # Formateo final de tensores
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    if edge_attr:
        # PyTorch Geometric espera que las features de enlaces sean floats
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        # Manejo de seguridad por si hay una molécula sin enlaces (ej. un ion suelto)
        # El tamaño del vector de enlaces de este código es de 6 dimensiones
        num_bond_features = len(BOND_FEATURES['bond_type']) + 1 + 1 # 4 + 1 + 1 = 6
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.float)

    # === 3. COORDENADAS 3D (Opcional) ===
    # Mantengo el pos por si tu GNN usa posiciones (ej. SchNet o EGNN)
    # Si tu GNN no las usa, puedes borrar este bloque.
    try:
        conf = mol.GetConformer()
        pos = []
        for atom in mol.GetAtoms():
            p = conf.GetAtomPosition(atom.GetIdx())
            pos.append([p.x, p.y, p.z])
        pos = torch.tensor(pos, dtype=torch.float)
    except ValueError:
        # Si la molécula viene de un SMILES sin conformación 3D generada
        pos = None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

def get_model_dimensions(mode='one_hot', atom_emb_dim=None, hibrid_emb_dim=None, bond_emb_dim=None):
    """
    Calcula las dimensiones de entrada para los nodos (input_dim) 
    y enlaces (edge_dim) según el modo de extracción de features.
    """
    if mode == 'one_hot':
        # Sumamos las longitudes de las listas en el diccionario ATOM_FEATURES
        # + 1 por el flag booleano (aromatic) que no usa one-hot
        input_dim = (
            len(ATOM_FEATURES['atomic_num']) +
            len(ATOM_FEATURES['formal_charge']) +
            len(ATOM_FEATURES['hybridization']) +
            1 +  # is_aromatic flag (float)
            len(ATOM_FEATURES['ring_size']) +
            len(ATOM_FEATURES['h_bonding'])
        )
        
        # Sumamos la longitud de bond_types
        # + 2 por los flags booleanos (is_conjugated e is_in_ring)
        edge_dim = (
            len(BOND_FEATURES['bond_type']) +
            1 +  # is_conjugated flag (float)
            1    # is_in_ring flag (float)
        )
        
    elif mode == 'embedding':
        # Tu lógica original para embeddings
        calc_atom_emb_dim = calc_dim(len(periodic_elements) * atom_emb_dim)
        calc_hibrid_emb_dim = calc_dim(len(hybridization_types) * hibrid_emb_dim)
        calc_bond_emb_dim = calc_dim(N_BOND_TYPES * bond_emb_dim)

        input_dim = calc_atom_emb_dim + calc_hibrid_emb_dim + OTHER_NODE_FEATURES
        edge_dim = calc_bond_emb_dim + OTHER_EDGE_FEATURES
        
    else:
        raise ValueError(f"Modo desconocido: {mode}. Usa 'one_hot' o 'embedding'.")
        
    return input_dim, edge_dim

def calc_dim(x):
    return max(1, math.ceil(x))

def onehot_to_indices(data):
    """
    Convierte features one-hot a indices.
    Soporta la detección de 'Zero Masking' asignando clases Unknown/Other.
    """
    if data.x is None: return data

    x = data.x.clone()
    
    # Usamos las longitudes dinámicas de las constantes
    num_atoms = len(periodic_elements)
    num_hybrids = len(hybridization_types)
    
    # Indices de fallback
    idx_unknown_atom = num_atoms - 1 
    idx_unknown_hybrid = num_hybrids - 1

    # === 1. ÁTOMOS ===
    atom_onehot = x[:, :num_atoms]
    atom_idx = atom_onehot.argmax(dim=1, keepdim=True)
    
    # Detección de Ceros (Masking) -> Unknown
    is_empty_atom = (atom_onehot.sum(dim=1, keepdim=True) == 0)
    atom_idx[is_empty_atom] = idx_unknown_atom
    atom_idx = atom_idx.float()

    # === 2. HIBRIDACIÓN ===
    hybrid_onehot = x[:, -num_hybrids:]
    hybrid_idx = hybrid_onehot.argmax(dim=1, keepdim=True)
    
    # Detección de Ceros -> Other
    is_empty_hybrid = (hybrid_onehot.sum(dim=1, keepdim=True) == 0)
    hybrid_idx[is_empty_hybrid] = idx_unknown_hybrid
    hybrid_idx = hybrid_idx.float()

    # === 3. Concatenar ===
    cont_features = x[:, num_atoms:-num_hybrids]
    x_new = torch.cat([atom_idx, hybrid_idx, cont_features], dim=1)
    
    data_new = data.clone()
    data_new.x = x_new

    # === 4. ENLACES (Actualizado con Rotación) ===
    # Ajustamos la condición: al menos 1 feature de bond + dist + bond_flexibility = 3
    if data_new.edge_attr is not None and data_new.edge_attr.shape[1] >= 3:
        edge_attr = data_new.edge_attr
        
        # Ahora tenemos DOS features continuas al final (dist y bond_flexibility)
        cont_features = edge_attr[:, -2:] 
        bond_onehot = edge_attr[:, :-2] # Todo menos las últimas dos columnas
        
        bond_idx = bond_onehot.argmax(dim=1, keepdim=True)
        
        # --- CORRECCIÓN PARA ENLACES ---
        idx_unknown_bond = UNKNOWN_BOND_IDX 
        
        is_empty_bond = (bond_onehot.sum(dim=1, keepdim=True) == 0)
        bond_idx[is_empty_bond] = idx_unknown_bond
        bond_idx = bond_idx.float()
        
        # Concatenamos el índice recuperado con la distancia y la rotación
        data_new.edge_attr = torch.cat([bond_idx, cont_features], dim=1)

    return data_new

def read_targets(targets_file):
    
    #Lee el archivo TXT de targets y devuelve un diccionario {mol_id: target}.
    
    target_dict = {}
    with open(targets_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            mol_id = parts[0]
            target = float(parts[1])
            target_dict[mol_id] = target
    return target_dict


def load_data_from_sdf(sdf_dir, target_dict):
    data_list = []
    for filename in sorted(os.listdir(sdf_dir)):
        if not filename.endswith('.sdf'):
            continue

        mol_id = os.path.splitext(filename)[0]

        # Quitar sufijos alternativos si el target no coincide exactamente
        if mol_id not in target_dict and "_" in mol_id:
            mol_id_alt = mol_id.split('_')[0]
            if mol_id_alt in target_dict:
                mol_id = mol_id_alt

        if mol_id not in target_dict:
            logger.warning(f"No se encontró target para '{mol_id}', se omite.")
            continue

        filepath = os.path.join(sdf_dir, filename)
        suppl = Chem.SDMolSupplier(filepath, removeHs=True)
        mol = next((m for m in suppl if m is not None), None)

        if mol is None:
            logger.warning(f"No se pudo leer la molécula desde '{filename}', se omite.")
            continue

        # Ignorar moléculas sin enlaces
        if mol.GetNumBonds() == 0:
            logger.warning(f"'{filename}' no tiene enlaces, se omite.")
            continue

        try:
            data = mol_to_graph_data(mol)
        except Exception as e:
            logger.warning(f"Error procesando '{filename}': {e}")
            continue

        data.y = torch.tensor([target_dict[mol_id]], dtype=torch.float)
        data.name = mol_id
        data_list.append(data)

    if not data_list:
        raise ValueError("No se pudo cargar ninguna molécula válida.")

    # ... después de obtener tu data_list
    # output_path = f"{RESULTADOS_DIR}/data_list_philippe.pt"
    # torch.save(data_list, output_path)
    # print(f"Dataset guardado con éxito en {output_path}")

    return data_list



def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)

def one_of_k_encoding_unk(value, choices):
    #Devuelve un vector one-hot para 'value', usando 'choices'. Si no está, se usa la última categoría (Unknown).
    encoding = [0] * len(choices)
    if value not in choices:
        value = choices[-1]  # Unknown
    encoding[choices.index(value)] = 1
    return encoding

def prepare_sdf_training_data(sdf_dir, target_file, batch_size=32, valid_split=0.2):
    """
    Prepara los datos de entrenamiento y validación a partir de un directorio SDF y archivo target.

    Devuelve:
        train_loader, val_loader, device, input_dim, edge_dim, targetname
    """
    # Leer targets
    target_dict = read_targets(target_file)
    targetname = os.path.splitext(os.path.basename(target_file))[0]

    # Cargar datos desde SDF
    data_list = load_data_from_sdf(sdf_dir, target_dict)

    # Dividir entrenamiento/validación
    if 0 < valid_split < 1:
        train_data, val_data = train_test_split(data_list, test_size=valid_split, random_state=42)
        val_loader = create_dataloader(val_data, batch_size=batch_size)
    else:
        train_data = data_list
        val_loader = None

    train_loader = create_dataloader(train_data, batch_size=batch_size)

    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return train_loader, val_loader, device, targetname

def prepare_split_training_data(train_dir, val_dir, target_file, batch_size=32):
    """
    Prepara los datos cargando explícitamente desde carpetas separadas 
    para entrenamiento y validación.

    Devuelve:
        train_loader, val_loader, device, targetname
    """
    # 1. Leer el diccionario general de targets
    # Asumimos que target_file tiene los valores para TODAS las moléculas (train y val)
    target_dict = read_targets(target_file)
    targetname = os.path.splitext(os.path.basename(target_file))[0]

    # 2. Cargar datos desde los directorios específicos
    print(f"Cargando dataset de Entrenamiento desde: {train_dir}")
    train_data = load_data_from_sdf(train_dir, target_dict)

    print(f"Cargando dataset de Validación desde: {val_dir}")
    val_data = load_data_from_sdf(val_dir, target_dict)

    # 3. Crear los DataLoaders directamente
    # Entrenamiento SIEMPRE debe mezclarse (shuffle=True) para que la red aprenda mejor
    train_loader = create_dataloader(train_data, batch_size=batch_size, shuffle=True)
    
    # Validación NO necesita mezclarse (shuffle=False) para que la evaluación sea determinista
    val_loader = create_dataloader(val_data, batch_size=batch_size, shuffle=False)

    # 4. Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Listos: {len(train_data)} moléculas de train | {len(val_data)} moléculas de val")

    return train_loader, val_loader, device, targetname

def prepare_pt_training_data(pt_file_path, batch_size=32, valid_split=0.2):
    """
    Prepara los datos de entrenamiento y validación directamente desde un archivo .pt.

    Devuelve:
        train_loader, val_loader, device, targetname
    """
    if not os.path.exists(pt_file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {pt_file_path}")

    print(f"Cargando dataset preprocesado desde: {pt_file_path}...")
    
    # 1. Cargar la lista de moléculas directamente
    data_list = torch.load(pt_file_path)

    if not data_list or not isinstance(data_list, list):
        raise ValueError("El archivo .pt está vacío o no contiene una lista válida.")

    # 2. Obtener un nombre base para tus logs/guardados (opcional)
    # Por ejemplo, si el archivo es "mis_datos_targetX.pt", targetname será "mis_datos_targetX"
    targetname = os.path.splitext(os.path.basename(pt_file_path))[0]

    # 3. Dividir en entrenamiento y validación
    if 0 < valid_split < 1:
        train_data, val_data = train_test_split(data_list, test_size=valid_split, random_state=42)
        val_loader = create_dataloader(val_data, batch_size=batch_size)
        print(f"Dataset dividido: {len(train_data)} train | {len(val_data)} val")
    else:
        train_data = data_list
        val_loader = None
        print(f"Dataset cargado completo: {len(train_data)} train (sin validación)")

    train_loader = create_dataloader(train_data, batch_size=batch_size)

    # 4. Configurar el dispositivo (GPU o CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo configurado: {device}")

    return train_loader, val_loader, device, targetname

def prepare_split_pt_training_data(train_pt_path, val_pt_path, batch_size=32):
    """
    Prepara los datos de entrenamiento y validación cargando directamente 
    desde dos archivos .pt separados.

    Devuelve:
        train_loader, val_loader, device, targetname
    """
    # 1. Validar que ambos archivos existen
    if not os.path.exists(train_pt_path):
        raise FileNotFoundError(f"No se encontró el archivo de train: {train_pt_path}")
    if not os.path.exists(val_pt_path):
        raise FileNotFoundError(f"No se encontró el archivo de validation: {val_pt_path}")

    # 2. Cargar las listas de moléculas directamente a memoria
    print(f"Cargando dataset de Entrenamiento desde: {train_pt_path}")
    train_data = torch.load(train_pt_path)
    
    print(f"Cargando dataset de Validación desde: {val_pt_path}")
    val_data = torch.load(val_pt_path)

    # Chequeo rápido de seguridad
    if not train_data or not isinstance(train_data, list):
        raise ValueError("El archivo .pt de entrenamiento está vacío o no es válido.")
    if not val_data or not isinstance(val_data, list):
        raise ValueError("El archivo .pt de validación está vacío o no es válido.")

    # 3. Obtener un nombre base para tus logs/guardados
    # Usamos el nombre del archivo de entrenamiento como base
    targetname = os.path.splitext(os.path.basename(train_pt_path))[0]

    # 4. Crear los DataLoaders
    # Entrenamiento SIEMPRE debe mezclarse (shuffle=True) para mejor aprendizaje
    train_loader = create_dataloader(train_data, batch_size=batch_size, shuffle=True)
    
    # Validación NO necesita mezclarse (shuffle=False) para evaluaciones consistentes
    val_loader = create_dataloader(val_data, batch_size=batch_size, shuffle=False)

    # 5. Configurar el dispositivo (GPU o CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Listos: {len(train_data)} grafos en Train | {len(val_data)} grafos en Val")
    print(f"Dispositivo configurado: {device}")

    return train_loader, val_loader, device, targetname