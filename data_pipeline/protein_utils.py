import torch
from data_pipeline.atom_utils import atom_type_onehot
from Bio.PDB import PDBParser


def protein_atom_features(atom_name):
    symbol = atom_name[0]
    return atom_type_onehot(symbol) + [0, 0]


def load_protein(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", pdb_path)
    coords, feats = {}, {}
    for atom in structure.get_atoms():
        res = atom.get_parent()
        resname = res.get_resname()
        resid = res.get_id()[1]
        atom_name = atom.get_name().strip()
        key = f"{resname}_{resid}_{atom_name}"
        coords[key] = torch.tensor(atom.coord, dtype=torch.float)
        feats[key] = torch.tensor(protein_atom_features(atom_name), dtype=torch.float)
    return coords, feats, structure


def filter_protein_atoms_by_distance(prot_coords, lig_coords, cutoff=5.0):
    if not prot_coords or not lig_coords:
        return {}
    prot_keys = list(prot_coords.keys())
    lig_positions = torch.stack([lig_coords[k] for k in lig_coords])
    filtered_keys = []
    for k in prot_keys:
        prot_pos = prot_coords[k].unsqueeze(0)  # shape [1,3]
        dist = torch.cdist(prot_pos, lig_positions)  # [1, #lig_atoms]
        if torch.min(dist) <= cutoff:
            filtered_keys.append(k)
    return {k: prot_coords[k] for k in filtered_keys}
