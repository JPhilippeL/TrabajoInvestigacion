import os
from time import time

import networkx as nx
import pandas as pd
import torch
from Bio.PDB import PDBParser
from rdkit import Chem
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
from tqdm import tqdm


# =========================================================
# pIC50 loader
# =========================================================


def load_pic50(pic50_path):
    df = pd.read_csv(
        pic50_path,
        sep=r"\s+|,|\t",
        engine="python",
        header=None,
        names=["pdb_id", "pIC50"],
    )
    return dict(zip(df["pdb_id"], df["pIC50"]))


# =========================================================
# Atomic feature helpers
# =========================================================
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


# =========================================================
# Coordinate and graph helpers
# =========================================================
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
    """Mantiene solo átomos de la proteína dentro de `cutoff` Å de algún átomo del ligando."""
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


def mol2graph(mol):
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        atom_feats = torch.tensor(ligand_atom_features(atom))
        graph.add_node(atom.GetIdx(), feats=atom_feats)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        graph.add_edge(i, j)
    graph = graph.to_directed()
    x = torch.stack([feats["feats"] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack(
        [torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]
    ).T
    return x, edge_index


def inter_graph(ligand_mol, protein_mol, pos_l, pos_p, dis_threshold=5.0):
    num_l, num_p = pos_l.shape[0], pos_p.shape[0]
    dist_mat = distance_matrix(pos_l, pos_p)
    idx = torch.nonzero(torch.tensor(dist_mat) < dis_threshold)
    edges = [[i, j + num_l] for i, j in idx]
    edges += [[j + num_l, i] for i, j in idx]  # bidirectional
    return (
        torch.tensor(edges).T if edges else torch.empty(2, 0, dtype=torch.long)
    )


# =========================================================
# Build graph compatible with CheapNet original (v04 FIXED)
# =========================================================
def build_graph_for_original(ligand_sdf_dir, proteine_pdb_dir, pdb_id, pic50_dict, dis_threshold=5.0, cutoff_prot=6.0):
    lig_coords, lig_feats, lig_mol = load_ligand(os.path.join(ligand_sdf_dir, f"{pdb_id}_ligand.sdf"))
    prot_coords, prot_feats, _ = load_protein(os.path.join(proteine_pdb_dir, f"{pdb_id}_protein.pdb"))

    if not lig_coords or not prot_coords:
        return None

    # -----------------------------------------------------
    # Filtrar proteína por proximidad al ligando
    # -----------------------------------------------------
    prot_coords = filter_protein_atoms_by_distance(
        prot_coords, lig_coords, cutoff=cutoff_prot
    )
    prot_feats = {k: prot_feats[k] for k in prot_coords}

    if not prot_coords:
        return None

    # -----------------------------------------------------
    # Construcción de nodos
    # -----------------------------------------------------
    nodes = {}
    ligand_nodes, protein_nodes = set(), set()
    i_counter = 0

    for k in lig_coords.keys():
        nodes[k] = i_counter
        ligand_nodes.add(k)
        i_counter += 1

    for k in prot_coords.keys():
        nodes[k] = i_counter
        protein_nodes.add(k)
        i_counter += 1

    # -----------------------------------------------------
    # Posiciones y features base
    # -----------------------------------------------------
    pos_l_tensor = torch.stack([lig_coords[k] for k in lig_coords])
    pos_p_tensor = torch.stack([prot_coords[k] for k in prot_coords])

    pos = torch.cat([pos_l_tensor, pos_p_tensor], dim=0)

    feat_list = [lig_feats[k] for k in lig_feats] + [
        prot_feats[k] for k in prot_feats
    ]
    x = torch.stack(feat_list)

    num_l = pos_l_tensor.shape[0]
    num_p = pos_p_tensor.shape[0]

    # -----------------------------------------------------
    # INTRA-LIGAND (enlaces químicos RDKit)
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # INTRA-PROTEIN (por distancia)
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # INTRA TOTAL
    # -----------------------------------------------------
    edge_index_intra = torch.cat(
        [edge_index_intra_lig, edge_index_intra_pro], dim=1
    )

    # -----------------------------------------------------
    # INTER (ligando-proteína por distancia)
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # EDGE TOTAL
    # -----------------------------------------------------
    edge_index_total = torch.cat([edge_index_intra, edge_index_inter], dim=1)

    # -----------------------------------------------------
    # Features geométricas (grado + distancia media)
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------
    y = torch.tensor([pic50_dict[pdb_id]], dtype=torch.float32)

    # -----------------------------------------------------
    # Split ligand/protein
    # -----------------------------------------------------
    split = torch.zeros(pos.shape[0], dtype=torch.long)
    split[:num_l] = 1  # ligand = 1, protein = 0

    # -----------------------------------------------------
    # Edge attributes
    # -----------------------------------------------------
    edge_attr = torch.norm(
        pos[edge_index_total[0]] - pos[edge_index_total[1]], dim=1, keepdim=True
    )

    # -----------------------------------------------------
    # Batch
    # -----------------------------------------------------
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


def generate_all_graphs(pic50_file, out_dir, ligand_sdf_dir, log_callback, proteine_pdb_dir, dis_threshold=5.0,
                        cutoff_prot=6.0):
    os.makedirs(out_dir, exist_ok=True)
    debut_generation = time()
    pic50_dict = load_pic50(pic50_file)
    for pdb_id in tqdm(pic50_dict.keys(), desc="Graph Generation CheapNet"):
        g = build_graph_for_original(
            ligand_sdf_dir,
            proteine_pdb_dir,
            pdb_id,
            pic50_dict,
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
