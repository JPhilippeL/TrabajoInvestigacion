#data_processing.py
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as TorchF
from ui.utils import periodic_elements, hybridization_types
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import logging
logger = logging.getLogger(__name__)

def mol_to_graph_data_obj(mol):
    bond_type_to_int = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }
    num_bond_types = len(bond_type_to_int)

    # === ÁTOMOS: usar vector completo ===
    atom_features = []
    for atom in mol.GetAtoms():
        features = one_of_k_encoding_unk(atom.GetSymbol(), periodic_elements) + \
                   [atom.GetDegree()/10.0] + \
                   [atom.GetTotalNumHs()/10.0] + \
                   [atom.GetIsAromatic()] + \
                   one_of_k_encoding_unk(atom.GetHybridization().name, hybridization_types)
        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float)

    # Coordenadas 3D
    conf = mol.GetConformer()
    pos = []
    for atom in mol.GetAtoms():
       idx = atom.GetIdx()
       p = conf.GetAtomPosition(idx)
       pos.append([p.x, p.y, p.z])
    pos = torch.tensor(pos, dtype=torch.float)

    # Indices y atributos de los enlaces
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Distancia Euclidiana
        dist = torch.norm(pos[i] - pos[j]).unsqueeze(0)  # tensor de tamaño [1]

        bond_type = bond.GetBondType()
        bond_type_idx = bond_type_to_int.get(bond_type, 0)  # default a 0
        bond_onehot = TorchF.one_hot(torch.tensor(bond_type_idx), num_classes=num_bond_types).float()

        # Concatenar tipo de enlace + distancia
        edge_features = torch.cat([bond_onehot, dist], dim=0)

        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(edge_features)
        edge_attr.append(edge_features)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr).float()

    # Construir objeto Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def mol_to_graph_data_obj_embedding(mol):
    # Mapeo de tipos de enlace a índices
    bond_type_to_int = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }

    # === ÁTOMOS ===
    atom_features = []
    atom_type_to_idx = {el: i for i, el in enumerate(periodic_elements)}
    hybrid_to_idx = {h: i for i, h in enumerate(hybridization_types)}

    for atom in mol.GetAtoms():
        symbol_idx = atom_type_to_idx.get(atom.GetSymbol(), len(atom_type_to_idx) - 1)
        hybrid_idx = hybrid_to_idx.get(atom.GetHybridization().name, len(hybrid_to_idx) - 1)
        degree = atom.GetDegree() / 10.0
        num_h = atom.GetTotalNumHs() / 10.0
        aromatic = float(atom.GetIsAromatic())

        # Guardamos los índices categóricos + valores continuos
        atom_features.append([symbol_idx, hybrid_idx, degree, num_h, aromatic])

    x = torch.tensor(atom_features, dtype=torch.float)

    # === Coordenadas 3D ===
    conf = mol.GetConformer()
    pos = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        p = conf.GetAtomPosition(idx)
        pos.append([p.x, p.y, p.z])
    pos = torch.tensor(pos, dtype=torch.float)

    # === ENLACES ===
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        dist = torch.norm(pos[i] - pos[j]).unsqueeze(0)  # tensor [1]
        bond_type = bond.GetBondType()
        bond_type_idx = bond_type_to_int.get(bond_type, 0)

        # Guardamos solo el índice y la distancia
        edge_features = torch.tensor([bond_type_idx, dist.item()], dtype=torch.float)

        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(edge_features)
        edge_attr.append(edge_features)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr).float()

    # === Construir objeto Data ===
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


# def smiles_to_graph_data_obj(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         raise ValueError(f"SMILES inválido: {smiles}")
    
#     # Añadir hidrógenos explícitos
#     mol = Chem.AddHs(mol)
    
#     # Generar coordenadas 3D
#     AllChem.EmbedMolecule(mol, randomSeed=42)
#     AllChem.UFFOptimizeMolecule(mol)
    
#     # Reutilizar
#     return mol_to_graph_data_obj(mol)

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
            data = mol_to_graph_data_obj_embedding(mol)
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

    # Dimensiones de entrada
    input_dim = data_list[0].x.shape[1]
    edge_dim = data_list[0].edge_attr.shape[1]

    return train_loader, val_loader, device, input_dim, edge_dim, targetname

# def process_csv(csv_file, valid_split, batch_size):
#     # Leer CSV
#     df = pd.read_csv(csv_file)
#     # Buscar columna SMILES (insensible a mayúsculas)
#     smiles_cols = [c for c in df.columns if c.lower() == "smiles"]
#     if not smiles_cols:
#         raise ValueError("El CSV debe contener una columna con SMILES (insensible a mayúsculas).")
#     smiles_col = smiles_cols[0]

#     # Buscar columna target (cualquier columna que contenga 'target', insensible a mayúsculas)
#     target_cols = [c for c in df.columns if "target" in c.lower()]
#     if not target_cols:
#         raise ValueError("El CSV debe contener al menos una columna que tenga 'target' en su nombre.")
#     target_col = target_cols[0]
#     safe_target_name = re.sub(r"[^A-Za-z0-9_\-]", "_", target_col)
        
#     data_list = []
#     for _, row in df.iterrows():
#         try:
#             graph_data = smiles_to_graph_data_obj(row[smiles_col])
#             graph_data.y = torch.tensor([row[target_col]], dtype=torch.float)
#             data_list.append(graph_data)
#         except Exception as e:
#             logging.warning(f"Error con SMILES {row[smiles_col]}: {e}")

#     logging.info(f"Se pudieron traducir correctamente {len(data_list)} de {len(df)} moléculas.")

#     if not data_list:
#         raise ValueError("No se pudo generar ningún grafo a partir del CSV")

#     # Dividir entrenamiento/validación
#     if 0 < valid_split < 1:
#         train_data, val_data = train_test_split(data_list, test_size=valid_split, random_state=42)
#         val_loader = create_dataloader(val_data, batch_size=batch_size)
#     else:
#         train_data = data_list
#         val_loader = None

#     train_loader = create_dataloader(train_data, batch_size=batch_size)
#     input_dim = data_list[0].x.shape[1]
#     edge_dim = data_list[0].edge_attr.shape[1]

#     return train_loader, val_loader, input_dim, edge_dim, safe_target_name

