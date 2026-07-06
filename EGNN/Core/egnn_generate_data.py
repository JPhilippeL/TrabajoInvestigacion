"""
@file egnn_generate_data.py
@author Mohamed EL BOUKHIARI
@brief Graph generation pipeline for the EGNN model.
@details
This file generates PyTorch Geometric graph files from the URV dataset.

It is adapted from the original 04_a_DB_Generation_EGNN.py script.
The original script logic is exposed through callable functions for GUI
integration.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from rdkit import Chem
from Bio.PDB import PDBParser

try:
    from EGNN.utils.constants import (
        DEFAULT_PIC50_FILE,
        DEFAULT_LIGAND_SDF_DIR,
        DEFAULT_PROTEIN_PDB_DIR,
        DEFAULT_GRAPHS_DIR,
        DEFAULT_CUTOFF_EDGES,
        DEFAULT_CUTOFF_PROT,
    )
except ImportError:
    # Fallback for direct script execution outside the package context.
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    MODULE_ROOT = os.path.dirname(PROJECT_ROOT)
    DATASET_DIR = os.path.join(MODULE_ROOT, "MPro-URV_Version2")
    DEFAULT_PIC50_FILE = os.path.join(DATASET_DIR, "pIC50.txt")
    DEFAULT_LIGAND_SDF_DIR = os.path.join(DATASET_DIR, "Ligand", "Ligand_SDF")
    DEFAULT_PROTEIN_PDB_DIR = os.path.join(DATASET_DIR, "Protein", "Protein_PDB")
    DEFAULT_GRAPHS_DIR = os.path.join(MODULE_ROOT, "Graphs_EGNN")
    DEFAULT_CUTOFF_EDGES = 5.0
    DEFAULT_CUTOFF_PROT = 6.0


PIC50_FILE = DEFAULT_PIC50_FILE
LIGAND_SDF_DIR = DEFAULT_LIGAND_SDF_DIR
PROTEIN_PDB_DIR = DEFAULT_PROTEIN_PDB_DIR
GRAPH_OUT_DIR = DEFAULT_GRAPHS_DIR


def load_pic50(pic50_path: str) -> Dict[str, float]:
    """
    @brief Load pIC50 values from the dataset text file.
    @param pic50_path Path to pIC50.txt.
    @return Dictionary mapping pdb_id to pIC50 value.
    """
    df = pd.read_csv(
        pic50_path,
        sep=r"\s+|,|\t",
        engine="python",
        header=None,
        names=["pdb_id", "pIC50"],
    )
    return dict(zip(df["pdb_id"].astype(str), df["pIC50"].astype(float)))


ATOM_TYPES = ["C", "N", "O", "S", "H", "P", "F", "Cl", "Br", "I"]


def atom_type_onehot(symbol: str) -> list[int]:
    """
    @brief Convert an atom symbol into a one-hot vector.
    @param symbol Atom symbol.
    @return One-hot encoded atom type vector.
    """
    vec = [0] * len(ATOM_TYPES)
    if symbol in ATOM_TYPES:
        vec[ATOM_TYPES.index(symbol)] = 1
    return vec


def ligand_atom_features(atom) -> list[float]:
    """
    @brief Build a feature vector for one ligand atom.
    @param atom RDKit atom object.
    @return Feature vector.
    """
    symbol = atom.GetSymbol()
    onehot = atom_type_onehot(symbol)
    charge = atom.GetFormalCharge()
    aromatic = int(atom.GetIsAromatic())
    return onehot + [charge, aromatic]


def protein_atom_features(atom_name: str) -> list[float]:
    """
    @brief Build a feature vector for one protein atom.
    @param atom_name Protein atom name.
    @return Feature vector.
    """
    symbol = atom_name[0]
    return atom_type_onehot(symbol) + [0, 0]


def load_ligand(sdf_path: str) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    @brief Load ligand coordinates and features from an SDF file.
    @param sdf_path Path to ligand SDF file.
    @return Ligand positions and ligand features. Returns (None, None) on failure.
    """
    if not os.path.exists(sdf_path):
        return None, None

    mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if mol is None or mol.GetNumConformers() == 0:
        return None, None

    conf = mol.GetConformer()

    coords = []
    feats = []

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
        feats.append(ligand_atom_features(atom))

    if not coords:
        return None, None

    return (
        torch.tensor(coords, dtype=torch.float),
        torch.tensor(feats, dtype=torch.float),
    )


def load_protein(pdb_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    @brief Load protein coordinates and features from a PDB file.
    @param pdb_path Path to protein PDB file.
    @return Protein positions and protein features.
    """
    if not os.path.exists(pdb_path):
        return torch.empty((0, 3), dtype=torch.float), torch.empty((0, 12), dtype=torch.float)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    coords = []
    feats = []

    for atom in structure.get_atoms():
        coords.append(atom.coord)
        feats.append(protein_atom_features(atom.get_name()))

    if not coords:
        return torch.empty((0, 3), dtype=torch.float), torch.empty((0, 12), dtype=torch.float)

    return (
        torch.tensor(coords, dtype=torch.float),
        torch.tensor(feats, dtype=torch.float),
    )


def filter_protein_atoms(
    prot_pos: torch.Tensor,
    prot_feat: torch.Tensor,
    lig_pos: torch.Tensor,
    cutoff: float = DEFAULT_CUTOFF_PROT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    @brief Keep only protein atoms within a given distance of the ligand.
    @param prot_pos Protein atom coordinates.
    @param prot_feat Protein atom features.
    @param lig_pos Ligand atom coordinates.
    @param cutoff Distance threshold in Angstrom.
    @return Filtered protein positions and features.
    """
    if prot_pos.shape[0] == 0:
        return prot_pos, prot_feat

    dist = torch.cdist(prot_pos, lig_pos)
    mask = torch.min(dist, dim=1).values < cutoff

    return prot_pos[mask], prot_feat[mask]


def build_graph(
    pdb_id: str,
    pic50_dict: Dict[str, float],
    cutoff_edges: float = DEFAULT_CUTOFF_EDGES,
    cutoff_prot: float = DEFAULT_CUTOFF_PROT,
) -> Optional[Data]:
    """
    @brief Build one PyTorch Geometric graph for a protein-ligand complex.
    @param pdb_id PDB identifier.
    @param pic50_dict Dictionary of pIC50 values.
    @param cutoff_edges Distance threshold for graph edges.
    @param cutoff_prot Cutoff used to filter protein atoms near the ligand.
    @return PyG Data object, or None if the complex cannot be processed.
    """
    lig_pos, lig_feat = load_ligand(
        os.path.join(LIGAND_SDF_DIR, f"{pdb_id}_ligand.sdf")
    )

    if lig_pos is None or lig_feat is None:
        return None

    prot_pos, prot_feat = load_protein(
        os.path.join(PROTEIN_PDB_DIR, f"{pdb_id}_protein.pdb")
    )

    prot_pos, prot_feat = filter_protein_atoms(
        prot_pos,
        prot_feat,
        lig_pos,
        cutoff=cutoff_prot,
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

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    y = torch.tensor([pic50_dict[pdb_id]], dtype=torch.float32)

    return Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        y=y,
    )


def generate_all_graphs(
    pic50_dict: Dict[str, float],
    cutoff_edges: float = DEFAULT_CUTOFF_EDGES,
    cutoff_prot: float = DEFAULT_CUTOFF_PROT,
) -> int:
    """
    @brief Generate all EGNN graphs from the dataset.
    @param pic50_dict Dictionary of pIC50 values.
    @param cutoff_edges Distance threshold for graph edges.
    @param cutoff_prot Cutoff used to filter protein atoms near the ligand.
    @return Number of successfully generated graphs.
    """
    generated_count = 0

    for pdb_id in tqdm(pic50_dict.keys(), desc="Generating EGNN graphs"):
        graph = build_graph(
            pdb_id,
            pic50_dict,
            cutoff_edges=cutoff_edges,
            cutoff_prot=cutoff_prot,
        )

        if graph is None:
            continue

        torch.save(graph, os.path.join(GRAPH_OUT_DIR, f"{pdb_id}.pt"))
        generated_count += 1

    return generated_count


def generate_data(
    pic50_file: str | None = None,
    ligand_sdf_dir: str | None = None,
    protein_pdb_dir: str | None = None,
    graphs_dir: str | None = None,
    cutoff_edges: float = DEFAULT_CUTOFF_EDGES,
    cutoff_prot: float = DEFAULT_CUTOFF_PROT,
) -> dict:
    """
    @brief Main callable function for EGNN graph generation.
    @param pic50_file Path to pIC50.txt.
    @param ligand_sdf_dir Directory containing ligand SDF files.
    @param protein_pdb_dir Directory containing protein PDB files.
    @param graphs_dir Output directory for generated graphs.
    @param cutoff_edges Edge cutoff used to build graph connectivity.
    @param cutoff_prot Cutoff used to filter protein atoms near the ligand.
    @return Summary dictionary for GUI or logging usage.
    """
    global PIC50_FILE, LIGAND_SDF_DIR, PROTEIN_PDB_DIR, GRAPH_OUT_DIR

    PIC50_FILE = pic50_file or DEFAULT_PIC50_FILE
    LIGAND_SDF_DIR = ligand_sdf_dir or DEFAULT_LIGAND_SDF_DIR
    PROTEIN_PDB_DIR = protein_pdb_dir or DEFAULT_PROTEIN_PDB_DIR
    GRAPH_OUT_DIR = graphs_dir or DEFAULT_GRAPHS_DIR

    os.makedirs(GRAPH_OUT_DIR, exist_ok=True)

    pic50_dict = load_pic50(PIC50_FILE)
    generated_graphs = generate_all_graphs(
        pic50_dict,
        cutoff_edges=cutoff_edges,
        cutoff_prot=cutoff_prot,
    )

    return {
        "output_dir": GRAPH_OUT_DIR,
        "total_items": len(pic50_dict),
        "generated_graphs": generated_graphs,
        "skipped_items": len(pic50_dict) - generated_graphs,
        "cutoff_edges": cutoff_edges,
        "cutoff_prot": cutoff_prot,
    }


if __name__ == "__main__":
    result = generate_data()
    print("\nEGNN graph generation completed.")
    print(result)
