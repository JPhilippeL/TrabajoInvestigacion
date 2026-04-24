"""
@file egnn_generate_data.py
@author Mohamed EL BOUKHIARI
@brief Graph generation pipeline for the EGNN model.
@details
This file generates PyTorch Geometric graph files from the URV dataset.

It is adapted from the original 04_a_DB_Generation_EGNN.py script.
The original logic should be preserved, but exposed through a callable
function for later GUI integration.
"""

from __future__ import annotations

import os
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from tqdm import tqdm
from rdkit import Chem
from Bio.PDB import PDBParser


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(PROJECT_ROOT)
DATASET_DIR = os.path.join(MODULE_ROOT, "MPro-URV_Version2")
PIC50_FILE = os.path.join(DATASET_DIR, "pIC50.txt")

LIGAND_SDF_DIR = os.path.join(DATASET_DIR, "Ligand", "Ligand_SDF")
PROTEIN_PDB_DIR = os.path.join(DATASET_DIR, "Protein", "Protein_PDB")

GRAPH_OUT_DIR = os.path.join(MODULE_ROOT, "Graphs_EGNN")
os.makedirs(GRAPH_OUT_DIR, exist_ok=True)


# ============================================================
# PIC50 LOADER
# ============================================================

def load_pic50(pic50_path: str) -> dict:
    """
    @brief Load pIC50 values from the dataset text file.
    @param pic50_path Path to pIC50.txt.
    @return Dictionary mapping pdb_id -> pIC50 value.
    """

    df = pd.read_csv(pic50_path, sep=r"\s+|,|\t", engine="python",
                     header=None, names=["pdb_id", "pIC50"])
    return dict(zip(df["pdb_id"], df["pIC50"]))


# ============================================================
# ATOMIC FEATURES
# ============================================================

ATOM_TYPES = ["C", "N", "O", "S", "H", "P", "F", "Cl", "Br", "I"]


def atom_type_onehot(symbol: str):
    """
    @brief Convert an atom symbol into a one-hot vector.
    @param symbol Atom symbol.
    @return One-hot encoded atom type vector.
    """

    vec = [0]*len(ATOM_TYPES)
    if symbol in ATOM_TYPES:
        vec[ATOM_TYPES.index(symbol)] = 1
    return vec


def ligand_atom_features(atom):
    """
    @brief Build feature vector for one ligand atom.
    @param atom RDKit atom object.
    @return Feature vector.
    """

    symbol = atom.GetSymbol()
    onehot = atom_type_onehot(symbol)
    charge = atom.GetFormalCharge()
    aromatic = int(atom.GetIsAromatic())
    return onehot + [charge, aromatic]


def protein_atom_features(atom_name: str):
    """
    @brief Build feature vector for one protein atom.
    @param atom_name Protein atom name.
    @return Feature vector.
    """

    symbol = atom_name[0]
    return atom_type_onehot(symbol) + [0,0]


# ============================================================
# LOAD LIGAND
# ============================================================

def load_ligand(sdf_path: str):
    """
    @brief Load ligand coordinates and features from an SDF file.
    @param sdf_path Path to ligand SDF file.
    @return Ligand positions and ligand features.
    """

    mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if mol is None:
        return None, None

    conf = mol.GetConformer()

    coords = []
    feats = []

    for atom in mol.GetAtoms():

        pos = conf.GetAtomPosition(atom.GetIdx())

        coords.append([pos.x,pos.y,pos.z])
        feats.append(ligand_atom_features(atom))

    return torch.tensor(coords,dtype=torch.float), \
           torch.tensor(feats,dtype=torch.float)


# ============================================================
# LOAD PROTEIN
# ============================================================

def load_protein(pdb_path: str):
    """
    @brief Load protein coordinates and features from a PDB file.
    @param pdb_path Path to protein PDB file.
    @return Protein positions and protein features.
    """

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", pdb_path)

    coords = []
    feats = []

    for atom in structure.get_atoms():

        coords.append(atom.coord)
        feats.append(protein_atom_features(atom.get_name()))

    return torch.tensor(coords,dtype=torch.float), \
           torch.tensor(feats,dtype=torch.float)


# ============================================================
# FILTER PROTEIN ATOMS NEAR LIGAND
# ============================================================

def filter_protein_atoms(prot_pos, prot_feat, lig_pos, cutoff: float = 6.0):
    """
    @brief Keep only protein atoms within a given distance of the ligand.
    @param prot_pos Protein atom coordinates.
    @param prot_feat Protein atom features.
    @param lig_pos Ligand atom coordinates.
    @param cutoff Distance threshold in Å.
    @return Filtered protein positions and features.
    """

    if prot_pos.shape[0] == 0:
        return prot_pos, prot_feat

    dist = torch.cdist(prot_pos, lig_pos)
    mask = torch.min(dist,dim=1).values < cutoff

    return prot_pos[mask], prot_feat[mask]


# ============================================================
# BUILD EGNN GRAPH
# ============================================================

def build_graph(pdb_id: str, pic50_dict: dict, cutoff_edges: float = 5.0, cutoff_prot: float = 6.0):
    """
    @brief Build one PyG graph for a protein-ligand complex.
    @param pdb_id PDB identifier.
    @param pic50_dict Dictionary of pIC50 values.
    @param cutoff_edges Distance threshold for graph edges.
    @param cutoff_prot Distance threshold for filtering protein atoms.
    @return PyG Data object.
    """

    lig_pos, lig_feat = load_ligand(
        os.path.join(LIGAND_SDF_DIR,f"{pdb_id}_ligand.sdf")
    )

    prot_pos, prot_feat = load_protein(
        os.path.join(PROTEIN_PDB_DIR,f"{pdb_id}_protein.pdb")
    )

    if lig_pos is None:
        return None

    prot_pos, prot_feat = filter_protein_atoms(
        prot_pos, prot_feat, lig_pos, cutoff=cutoff_prot
    )

    if prot_pos.shape[0] == 0:
        return None

    pos = torch.cat([lig_pos, prot_pos], dim=0)
    x = torch.cat([lig_feat, prot_feat], dim=0)

    dist = torch.cdist(pos, pos)

    edges = []
    for i in range(pos.shape[0]):
        for j in range(pos.shape[0]):
            if i != j and dist[i, j] < cutoff_edges:
                edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).T

    y = torch.tensor([pic50_dict[pdb_id]],dtype=torch.float32)

    return Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        y=y
    )


# ============================================================
# GENERATE ALL GRAPHS
# ============================================================

def generate_all_graphs(pic50_dict: dict) -> int:
    """
    @brief Generate all EGNN graphs from the dataset.
    @param pic50_dict Dictionary of pIC50 values.
    @return Number of successfully generated graphs.
    """

    for pdb_id in tqdm(pic50_dict.keys(), desc="Generando grafos EGNN"):

        g = build_graph(pdb_id, pic50_dict)

        if g is None:
            continue

        torch.save(g, os.path.join(GRAPH_OUT_DIR,f"{pdb_id}.pt"))


def generate_data(
    pic50_file: str,
    ligand_sdf_dir: str,
    protein_pdb_dir: str,
    graphs_dir: str,
    cutoff_edges: float = 5.0,
    cutoff_prot: float = 6.0,
) -> dict:
    """
    @brief Main callable function for EGNN graph generation.
    @param pic50_file Path to pIC50.txt.
    @param ligand_sdf_dir Directory containing ligand SDF files.
    @param protein_pdb_dir Directory containing protein PDB files.
    @param graphs_dir Output directory for generated graphs.
    @param cutoff_edges Edge cutoff used to build graph connectivity.
    @param cutoff_prot Cutoff used to filter protein atoms near ligand.
    @return Summary dictionary for GUI or logging usage.
    """
    global PIC50_FILE, LIGAND_SDF_DIR, PROTEIN_PDB_DIR, GRAPH_OUT_DIR

    PIC50_FILE = pic50_file
    LIGAND_SDF_DIR = ligand_sdf_dir
    PROTEIN_PDB_DIR = protein_pdb_dir
    GRAPH_OUT_DIR = graphs_dir

    os.makedirs(GRAPH_OUT_DIR, exist_ok=True)

    pic50_dict = load_pic50(PIC50_FILE)
    total_graphs = generate_all_graphs(pic50_dict)

    return {
        "output_dir": graphs_dir,
        "total_graphs": total_graphs,
    }


if __name__ == "__main__":
    results = generate_data()
    print("\nGeneración de grafos EGNN completada.")
    print(results)
