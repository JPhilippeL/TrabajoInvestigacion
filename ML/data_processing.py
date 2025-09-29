#data_processing.py
import os
from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def mol_to_graph_data_obj(mol):
    bond_type_to_int = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }
    #Convierte una molécula RDKit a un objeto Data de PyTorch Geometric.
    
    # Elementos de los átomos
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum()])
    x = torch.tensor(atom_features, dtype=torch.float)

    # Indices y atributos de los enlaces
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_index.append([i, j])
        edge_index.append([j, i])

        bond_type = bond.GetBondType()
        bond_type_idx = bond_type_to_int.get(bond_type, 0)  # default a 0

        edge_attr.append([bond_type_idx])
        edge_attr.append([bond_type_idx])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Coordenadas 3D
    conf = mol.GetConformer()
    pos = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        p = conf.GetAtomPosition(idx)
        pos.append([p.x, p.y, p.z])
    pos = torch.tensor(pos, dtype=torch.float)

    # Construir objeto Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    return data


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
        
        mol_id = os.path.splitext(filename)[0]  # nombre completo del SDF

        # Quitar "_Ligand" solo si no existe exactamente en target_dict
        if mol_id not in target_dict and "_" in mol_id:
            mol_id_alt = mol_id.split('_')[0]
            if mol_id_alt in target_dict:
                mol_id = mol_id_alt

        if mol_id not in target_dict:
            print(f"Warning: No se encontró target para la molécula '{mol_id}', se omite.")
            continue

        affinity = target_dict[mol_id]
        filepath = os.path.join(sdf_dir, filename)
        suppl = Chem.SDMolSupplier(filepath, removeHs=False)
        mol = next((m for m in suppl if m is not None), None)
        if mol is None:
            print(f"Warning: No se pudo leer la molécula del archivo '{filename}', se omite.")
            continue

        data = mol_to_graph_data_obj(mol)
        data.y = torch.tensor([affinity], dtype=torch.float)
        data.name = mol_id  # nombre limpio que coincide con target

        data_list.append(data)

    if not data_list:
        raise ValueError("No se pudo cargar ninguna molécula válida.")

    return data_list



def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)
