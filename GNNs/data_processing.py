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
    ATOM_TYPE_TO_IDX, HYBRID_TO_IDX
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

def get_atom_features(atom, is_donor, is_acceptor, mode='one_hot'):
    # 1. Features básicas
    degree = atom.GetDegree() / 10.0
    num_h = atom.GetTotalNumHs() / 10.0
    is_aromatic = float(atom.GetIsAromatic())
    
    # 2. Cargas (Formal y Gasteiger)
    formal_charge = float(atom.GetFormalCharge())
    try:
        gasteiger_charge = atom.GetDoubleProp('_GasteigerCharge')
        if math.isnan(gasteiger_charge) or math.isinf(gasteiger_charge):
            gasteiger_charge = 0.0
    except KeyError:
        gasteiger_charge = 0.0
        
    # 3. H-Bonds (Convertimos bool a float 1.0/0.0)
    is_donor_feat = float(is_donor)
    is_acceptor_feat = float(is_acceptor)

    if mode == 'one_hot':
        return (one_of_k_encoding_unk(atom.GetSymbol(), periodic_elements) + 
                [degree, num_h, is_aromatic] + #, formal_charge, gasteiger_charge] + 
                [is_donor_feat, is_acceptor_feat] +  # <--- NUEVO
                one_of_k_encoding_unk(atom.GetHybridization().name, hybridization_types))
    
    elif mode == 'embedding':
        symbol_idx = ATOM_TYPE_TO_IDX.get(atom.GetSymbol(), UNKNOWN_ATOM_IDX)
        hybrid_idx = HYBRID_TO_IDX.get(atom.GetHybridization().name, UNKNOWN_HYBRID_IDX)
        
        # Añadimos al final de las features continuas
        return [symbol_idx, hybrid_idx, degree, num_h, is_aromatic,
                is_donor_feat, is_acceptor_feat] 
                # formal_charge, gasteiger_charge, is_donor_feat, is_acceptor_feat]
    
    else:
        raise ValueError(f"Modo desconocido: {mode}")

def get_edge_features(bond_type_idx, dist, num_bond_types, mode='one_hot'):
    """Construye el vector de características del enlace."""
    
    if mode == 'one_hot':
        # One-hot encoding del tipo de enlace + distancia
        bond_onehot = F.one_hot(torch.tensor(bond_type_idx), num_classes=num_bond_types).float()
        return torch.cat([bond_onehot, dist], dim=0)
        
    elif mode == 'embedding':
        # Índice del tipo de enlace + distancia
        return torch.tensor([bond_type_idx, dist.item()], dtype=torch.float)

def mol_to_graph_data(mol, mode='embedding'):
    """
    Función unificada para convertir molécula a data
    Args:
        mol: Objeto molécula de RDKit.
        mode: 'one_hot' (para el primer caso) o 'embedding' (para el segundo).
    """
    
    # A. Cargas Gasteiger
    AllChem.ComputeGasteigerCharges(mol)
    
    # B. Identificar Donadores y Aceptores (Indices)
    # GetSubstructMatches devuelve tuplas de tuplas ((idx1,), (idx2,), ...), lo aplanamos a un set
    hbd_matches = mol.GetSubstructMatches(HBD_PATTERN)
    hbd_indices = {idx[0] for idx in hbd_matches} # Usamos set para búsqueda rápida O(1)

    hba_matches = mol.GetSubstructMatches(HBA_PATTERN)
    hba_indices = {idx[0] for idx in hba_matches}

    # === 1. ÁTOMOS ===
    atom_features = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        
        # Chequeamos si el índice actual está en los sets calculados
        is_donor = idx in hbd_indices
        is_acceptor = idx in hba_indices
        
        # Pasamos los flags a la función auxiliar
        atom_features.append(get_atom_features(atom, is_donor, is_acceptor, mode=mode))

    x = torch.tensor(atom_features, dtype=torch.float)

    # === 2. COORDENADAS 3D ===
    # Esta parte es idéntica en ambas funciones
    conf = mol.GetConformer()
    pos = []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        pos.append([p.x, p.y, p.z])
    pos = torch.tensor(pos, dtype=torch.float)

    # === 3. ENLACES ===
    edge_index = []
    edge_attr = []
    num_bond_types = len(BOND_TYPE_TO_INT)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Calcular distancia
        dist = torch.norm(pos[i] - pos[j]).unsqueeze(0)

        # Tipo de enlace
        bond_type_idx = BOND_TYPE_TO_INT.get(bond.GetBondType(), UNKNOWN_BOND_IDX )

        # Obtener features del enlace (delegado a función auxiliar)
        edge_features = get_edge_features(bond_type_idx, dist, num_bond_types, mode=mode)

        # Grafo no dirigido: agregamos (i, j) y (j, i)
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(edge_features)
        edge_attr.append(edge_features)

    # Formateo final de tensores
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    if edge_attr: # Prevenir error si no hay enlaces
        edge_attr = torch.stack(edge_attr).float()
    else:
        edge_attr = torch.empty((0, x.size(1))) # O dimensión adecuada vacía

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

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

    # === 4. ENLACES (Actualizado) ===
    if data_new.edge_attr is not None and data_new.edge_attr.shape[1] > 2:
        edge_attr = data_new.edge_attr
        
        dist = edge_attr[:, -1].unsqueeze(1)
        bond_onehot = edge_attr[:, :-1] # Todo menos distancia
        
        bond_idx = bond_onehot.argmax(dim=1, keepdim=True)
        
        # --- CORRECCIÓN PARA ENLACES ---
        # Si borramos el enlace (todo ceros), asignamos UNKNOWN_BOND_IDX
        # Esto es vital para que GraphExplainer funcione bien con enlaces.
        idx_unknown_bond = UNKNOWN_BOND_IDX 
        
        is_empty_bond = (bond_onehot.sum(dim=1, keepdim=True) == 0)
        bond_idx[is_empty_bond] = idx_unknown_bond
        bond_idx = bond_idx.float()
        
        data_new.edge_attr = torch.cat([bond_idx, dist], dim=1)

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
        suppl = Chem.SDMolSupplier(filepath, removeHs=False)
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