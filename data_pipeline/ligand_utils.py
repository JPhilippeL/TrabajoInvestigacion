import torch
from data_pipeline.atom_utils import atom_type_onehot
from rdkit import Chem


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
