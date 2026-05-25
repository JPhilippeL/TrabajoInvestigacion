import os

import networkx as nx
import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser, PPBuilder
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
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


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
    'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
    'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
    'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
    'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
]


def atom_features(atom):
    features = np.array(one_of_k_encoding_unk(atom.GetSymbol(), ATOM_TYPES) +
                        one_of_k_encoding(atom.GetDegree(), [i for i in range(0, 11, 1)]) +
                        one_of_k_encoding_unk(atom.GetTotalNumHs(), [i for i in range(0, 11, 1)]) +
                        one_of_k_encoding_unk(atom.GetImplicitValence(), [i for i in range(0, 11, 1)]) +
                        [atom.GetIsAromatic()], dtype=np.float32)

    feature_sum = features.sum()

    if feature_sum != 0:
        features /= feature_sum

    return features


# Ligand
def sdf_to_graphs(sdf_file_path):
    mol = Chem.MolFromMolFile(sdf_file_path, removeHs=False)
    if mol is None: return None

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


# Proteine
def load_protein_sequence(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", pdb_path)
    ppb = PPBuilder()
    sequences = []
    for peptide in ppb.build_peptides(structure):
        sequences.append(str(peptide.get_sequence()))

    return "".join(sequences)


def encode_proteine_sequence():
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    return seq_dict


def seq_cat(prot, max_len_sequence):
    x = np.zeros(max_len_sequence, dtype=np.int64)
    seq_dict = encode_proteine_sequence()
    for i, ch in enumerate(prot[:max_len_sequence]):
        x[i] = seq_dict.get(ch, 0)
    return x


# build data
def build_dta_data(pdb_id, pic50, sdf_dir, pdb_dir):
    sdf_path = os.path.join(sdf_dir, f"{pdb_id}_ligand.sdf")
    pdb_path = os.path.join(pdb_dir, f"{pdb_id}_protein.pdb")

    if not os.path.isfile(sdf_path):
        print(f"[ERROR GRAPHDTA LIGAND FILE]: Missing file {sdf_path}")
        return None

    if not os.path.isfile(pdb_path):
        print(f"[ERROR GRAPHDTA PROTEIN FILE]: Missing file {pdb_path}")
        return None

    ligand_graph = sdf_to_graphs(sdf_path)
    if ligand_graph is None:
        print(f"[ERROR GRAPHDTA LIGAND PARSE]:Could not parse the {sdf_path}")
        return None

    x, edge_index = ligand_graph

    proteine_sequence = load_protein_sequence(pdb_path)

    if len(proteine_sequence) == 0:
        print(f"[ERROR GRAPHDTA PROTEIN PARSE]:Could not parse the {pdb_path}")
        return None

    target = seq_cat(proteine_sequence, max_len_sequence=1000)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([float(pic50)], dtype=torch.float32)
    )
    data.target = torch.tensor(target, dtype=torch.long).unsqueeze(0)
    data.pdb_id = pdb_id

    return data


def generate_all_dta_dta(output_dir, pic50_file, sdf_dir, pdb_dir):
    os.makedirs(output_dir, exist_ok=True)
    pic50_dict = load_pic50(pic50_file)
    for pdb_id, pic50 in tqdm(pic50_dict.items(), desc="GRAPH DTA data generation"):
        data = build_dta_data(pdb_id, pic50, sdf_dir, pdb_dir)

        if data is None:
            continue
        out_path = os.path.join(output_dir, f"{pdb_id}.pt")
        torch.save(data, out_path)


def debug_graph(file_pt):
    data = torch.load(file_pt, weights_only=False)

    print(f"Data for {data.pdb_id}:")
    print(f"  - Number of atoms: {data.x.shape[0]}")
    print(f"  - Atom feature dimension: {data.x.shape[1]}")
    print(f"  - Number of edges: {data.edge_index.shape[1]}")
    print(f"  - Target sequence length: {data.target.shape[1]}")
    print(f"  - Affinity (pIC50): {data.y.item()}")

    G = nx.Graph()

    num_nodes = data.x.shape[0]
    G.add_nodes_from(range(num_nodes))

    edge_list = data.edge_index.t().cpu().numpy()
    G.add_edges_from(edge_list)

    plt.figure(figsize=(10, 10))
    nx.draw(
        G,
        with_labels=True,
        node_color="skyblue",
        node_size=500,
        font_size=8
    )
    plt.title(f"Graph for {data.pdb_id}")
    plt.show()
