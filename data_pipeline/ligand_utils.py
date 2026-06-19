import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from data_pipeline.atom_utils import atom_features, atom_type_onehot


def ligand_atom_features(atom):
    symbol = atom.GetSymbol()
    onehot = atom_type_onehot(symbol)
    charge = atom.GetFormalCharge()
    aromatic = int(atom.GetIsAromatic())
    return onehot + [charge, aromatic]


def load_ligand(sdf_path):
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
    if mol is None:
        return {}, {}, None
    conf = mol.GetConformer()
    coords, feats = {}, {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        key = atom.GetSymbol() + str(idx)
        coords[key] = torch.tensor([pos.x, pos.y, pos.z], dtype=torch.float)
        feats[key] = torch.tensor(ligand_atom_features(atom), dtype=torch.float)
    return coords, feats, mol


def ligand_graph_dta(sdf_file_path):
    mol = Chem.MolFromMolFile(str(sdf_file_path), removeHs=False)
    if mol is None:
        return None

    if not mol.GetConformers():
        AllChem.EmbedMolecule(mol, randomSeed=42)
    features = []
    for atom in mol.GetAtoms():
        features.append(atom_features(atom))

    x = torch.tensor(np.array(features), dtype=torch.float)

    edges = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edges.append([i, j])
        edges.append([j, i])

    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return x, edge_index
