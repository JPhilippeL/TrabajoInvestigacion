import os
from time import time

import pandas as pd
import torch
from Bio.PDB import PDBParser
from rdkit import Chem
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
from tqdm import tqdm


def load_pic50(pic50_path):
    df = pd.read_csv(
        pic50_path,
        sep=r"\s+|,|\t",
        engine="python",
        header=None,
        names=["pdb_id", "pIC50"],
    )
    return dict(zip(df["pdb_id"], df["pIC50"]))


ATOM_TYPES = ["C", "N", "O", "S", "H", "P", "F", "Cl", "Br", "I"]


def atom_type_onehot(symbol):
    vec = [0] * len(ATOM_TYPES)
    if symbol in ATOM_TYPES:
        vec[ATOM_TYPES.index(symbol)] = 1
    return vec


def ligand_atom_features(atom):
    symbol = atom.GetSymbol()
    onehot = atom_type_onehot(symbol)
    charge = atom.GetFormalCharge()
    aromatic = int(atom.GetIsAromatic())
    return onehot + [charge, aromatic]


def protein_atom_features(atom_name):
    symbol = atom_name[0]
    return atom_type_onehot(symbol) + [0, 0]


def load_ligand(sdf_path):
    mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
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
        feats[key] = torch.tensor(
            protein_atom_features(atom_name), dtype=torch.float
        )
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


def build_graph_for_original(
        pdb_id, pic50_dict, lig_dir, pdb_dir, dis_threshold=5.0, cutoff_prot=6.0
):
    lig_coords, lig_feats, lig_mol = load_ligand(
        os.path.join(lig_dir, f"{pdb_id}_ligand.sdf")
    )
    prot_coords, prot_feats, _ = load_protein(
        os.path.join(pdb_dir, f"{pdb_id}_protein.pdb")
    )

    if not lig_coords or not prot_coords:
        return None

    prot_coords = filter_protein_atoms_by_distance(
        prot_coords, lig_coords, cutoff=cutoff_prot
    )
    prot_feats = {k: prot_feats[k] for k in prot_coords}

    if not prot_coords:
        return None

    pos_l_tensor = torch.stack([lig_coords[k] for k in lig_coords])
    pos_p_tensor = torch.stack([prot_coords[k] for k in prot_coords])

    pos = torch.cat([pos_l_tensor, pos_p_tensor], dim=0)

    feat_list = [lig_feats[k] for k in lig_feats] + [
        prot_feats[k] for k in prot_feats
    ]
    x = torch.stack(feat_list)

    num_l = pos_l_tensor.shape[0]
    num_p = pos_p_tensor.shape[0]

    edge_l = []
    for bond in lig_mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_l.append([i, j])
        edge_l.append([j, i])

    edge_index_intra_lig = (
        torch.tensor(edge_l, dtype=torch.long).T
        if edge_l
        else torch.empty(2, 0, dtype=torch.long)
    )

    dist_pp = distance_matrix(pos_p_tensor.numpy(), pos_p_tensor.numpy())

    edges_pro = []
    for i in range(num_p):
        for j in range(num_p):
            if i != j and dist_pp[i, j] < dis_threshold:
                edges_pro.append([i + num_l, j + num_l])

    edge_index_intra_pro = (
        torch.tensor(edges_pro, dtype=torch.long).T
        if edges_pro
        else torch.empty(2, 0, dtype=torch.long)
    )

    edge_index_intra = torch.cat(
        [edge_index_intra_lig, edge_index_intra_pro], dim=1
    )

    dist_lp = distance_matrix(pos_l_tensor.numpy(), pos_p_tensor.numpy())

    edges_inter = []
    for i in range(num_l):
        for j in range(num_p):
            if dist_lp[i, j] < dis_threshold:
                edges_inter.append([i, j + num_l])
                edges_inter.append([j + num_l, i])

    edge_index_inter = (
        torch.tensor(edges_inter, dtype=torch.long).T
        if edges_inter
        else torch.empty(2, 0, dtype=torch.long)
    )

    edge_index_total = torch.cat([edge_index_intra, edge_index_inter], dim=1)

    deg = (
        torch.bincount(edge_index_total[0], minlength=pos.shape[0])
        .float()
        .unsqueeze(1)
    )

    dist_feat = torch.zeros(pos.shape[0], 1)
    for i, j in edge_index_total.t():
        dist_feat[i] += torch.norm(pos[i] - pos[j])

    dist_feat /= torch.clamp(deg, min=1)

    x = torch.cat([x, deg, dist_feat], dim=1)

    y = torch.tensor([pic50_dict[pdb_id]], dtype=torch.float32)

    split = torch.zeros(pos.shape[0], dtype=torch.long)
    split[:num_l] = 1  # ligand = 1, protein = 0

    edge_attr = torch.norm(
        pos[edge_index_total[0]] - pos[edge_index_total[1]], dim=1, keepdim=True
    )

    batch = torch.zeros(pos.shape[0], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index_total,
        edge_attr=edge_attr,
        pos=pos,
        y=y,
        split=split,
        edge_index_intra=edge_index_intra,
        edge_index_inter=edge_index_inter,
        edge_index_intra_lig=edge_index_intra_lig,
        edge_index_intra_pro=edge_index_intra_pro,
        batch=batch,
    )


def generate_all_graphs(
        pic50_file, out_dir, lig_dir, pdb_dir, log_callback, dis_threshold=5.0, cutoff_prot=6.0
):
    os.makedirs(out_dir, exist_ok=True)
    debut_generation = time()
    pic50_dict = load_pic50(pic50_file)
    for pdb_id in tqdm(pic50_dict.keys(), desc="Generation GIGN's graph"):
        g = build_graph_for_original(
            pdb_id,
            pic50_dict,
            lig_dir,
            pdb_dir,
            dis_threshold=dis_threshold,
            cutoff_prot=cutoff_prot,
        )
        if g is None:
            continue

        torch.save(g, os.path.join(out_dir, f"{pdb_id}.pt"))
    end_generation = time()
    if log_callback:
        log_callback.info(f"Graph generation took {end_generation - debut_generation:.2f} seconds.")
        log_callback.info(f"Graph generation completed. Graphs saved in: {out_dir}")
