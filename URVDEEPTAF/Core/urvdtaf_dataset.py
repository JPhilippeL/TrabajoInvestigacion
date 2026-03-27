from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Dict, Optional, Union, Tuple, Any
from torch_geometric.data import Data, Batch
import os

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}

CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)
PT_FEATURE_SIZE = 40

# Atomic features for GNN
ATOM_FEATURES = {
    'atomic_num': [1, 6, 7, 8, 9, 15, 16, 17, 35, 53],  # Values explained in: prepare/utils/feature_map.py
    'formal_charge': [-1, 0, 1],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3
    ],
    'aromatic': [0, 1],
    'ring_size': [0, 3, 4, 5, 6, 7, 8],  # 0 means not in a ring
    'h_bonding': [0, 1, 2]  # 0: neither, 1: acceptor, 2: donor
}

# Total length of the one‐hot atom feature vector
GNN_NODE_FEATURE_SIZE = (
    len(ATOM_FEATURES['atomic_num'])
  + len(ATOM_FEATURES['formal_charge'])
  + len(ATOM_FEATURES['hybridization'])
  + 1  # aromatic flag
  + len(ATOM_FEATURES['ring_size'])
  + len(ATOM_FEATURES['h_bonding'])
)

# Bond features for GNN
BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'is_conjugated': [0, 1],
    'is_in_ring': [0, 1]
}

def one_hot_encoding(value: Any, feature_list: List) -> List[int]:
    """
    Create one-hot encoding for a value based on feature list.
    
    Args:
        value: The value to encode
        feature_list: List of possible values
        
    Returns:
        One-hot encoded list
    """
    if value not in feature_list:
        encoding = [0] * len(feature_list)
    else:
        encoding = [0] * len(feature_list)
        encoding[feature_list.index(value)] = 1
    return encoding

def get_atom_features(atom: Chem.Atom) -> List[int]:
    """
    Get atom features as a one-hot encoded vector.
    
    Args:
        atom: RDKit atom object
        
    Returns:
        Feature vector for atom
    """
    features = []
    
    # Atomic number
    features += one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    
    # Formal charge
    features += one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    
    # Hybridization
    features += one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    
    # Aromaticity
    features += [int(atom.GetIsAromatic())]
    
    # Ring size (RDKit ≥2023)
    ring_size = 0
    mol = atom.GetOwningMol()
    for ring in mol.GetRingInfo().AtomRings():
        if atom.GetIdx() in ring:
            ring_size = len(ring)
            break
    features += one_hot_encoding(ring_size, ATOM_FEATURES['ring_size'])
    
    # H-bonding
    h_bonding = 0
    if atom.GetAtomicNum() in [7, 8]:  # N or O
        h_bonding = 1  # acceptor
    if atom.GetTotalNumHs() > 0 and atom.GetAtomicNum() in [7, 8]:  # N–H or O–H
        h_bonding = 2  # donor
    features += one_hot_encoding(h_bonding, ATOM_FEATURES['h_bonding'])
    
    return features


def get_bond_features(bond: Chem.Bond) -> List[int]:
    """
    Get bond features as a one-hot encoded vector.
    
    Args:
        bond: RDKit bond object
        
    Returns:
        Feature vector for bond
    """
    features = []
    
    # Bond type
    features += one_hot_encoding(bond.GetBondType(), BOND_FEATURES['bond_type'])
    
    # Conjugation
    features += [int(bond.GetIsConjugated())]
    
    # Ring membership
    features += [int(bond.IsInRing())]
    
    return features

def smiles_to_graph(smiles: str) -> Optional[Data]:
    """
    Convert SMILES string to a molecular graph representation.
    
    Args:
        smiles: SMILES string
        
    Returns:
        PyTorch Geometric Data object or None if parsing fails
    """
    try:
        # Parse SMILES and get molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add hydrogen atoms
        mol = Chem.AddHs(mol)
        
        # Calculate 3D coordinates (for potential future use)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        
        # Get atom features
        num_atoms = mol.GetNumAtoms()
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(get_atom_features(atom))
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Get edge indices and features
        edge_indices = []
        edge_attributes = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add edges in both directions
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            
            # Add same features for both directions
            edge_attributes.append(get_bond_features(bond))
            edge_attributes.append(get_bond_features(bond))
        
        if len(edge_indices) == 0:  # Molecule with no bonds
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, len(get_bond_features(None))), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_atoms
        )
        
        return data
    
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None

def label_smiles(line: str, max_smi_len: int) -> np.ndarray:
    """Convert SMILES string to numeric representation."""
    X = np.zeros(max_smi_len, dtype=np.int64)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET.get(ch, 0) - 1  # Use 0 for unknown characters

    return X

class MyDataset(Dataset):
    def __init__(self, 
                 data_path: Union[str, Path], 
                 phase: str, 
                 max_seq_len: int, 
                 max_pkt_len: int, 
                 max_smi_len: int,
                 use_gnn: bool = False,
                 ligand_or_pocket: Optional[str] = None,
                 pkt_window: Optional[int] = None, 
                 pkt_stride: Optional[int] = None ):
        """
        Dataset class for DeepDTAF model.
        
        Args:
            data_path: Path to data directory
            phase: One of 'training', 'validation', or 'test'
            max_seq_len: Maximum sequence length
            max_pkt_len: Maximum pocket length
            max_smi_len: Maximum SMILES length
            use_gnn: Whether to use GNN for ligand representation
            pkt_window: Window size for pocket folding (optional)
            pkt_stride: Stride size for pocket folding (optional)
        """
        data_path = Path(data_path)
        self.use_gnn = use_gnn

        # Load affinity data
        affinity: Dict[str, float] = {}
        affinity_df = pd.read_csv(os.path.join(os.path.dirname(data_path), 'pIC50.txt'),   header=None, sep=r'\s+')
        for _, row in affinity_df.iterrows():
            affinity[row[0]] = row[1]
        self.affinity = affinity

        # Load ligand SMILES
        ligands_df = pd.read_csv(data_path / f"{phase}_smi.csv")
        ligands = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
        self.smi = ligands
        self.max_smi_len = max_smi_len
        
        # Pre-compute molecular graphs if using GNN
        if use_gnn:
            data_list = torch.load(os.path.join(os.path.dirname(data_path), f"data_list_ligand.pt"), weights_only=False)
            data_dict = {}
            for data in data_list:
                x = data.x.detach() if data.x is not None else None
                edge_attr = data.edge_attr.detach() if data.edge_attr is not None else None

                fixed_data = Data(
                    x=x,
                    edge_index=data.edge_index,
                    edge_attr=edge_attr,
                    y=data.y,  # targets can remain
                    num_nodes=data.num_nodes,
                    name=getattr(data, "name", None)
                )

                if fixed_data.name is None:
                    raise ValueError("Data object has no name. Each entry must have a unique name.")

                data_dict[fixed_data.name] = fixed_data

            #data_dict = {data.name: data for data in data_list}
            self.graphs = {}
            for pdbid, smiles in ligands.items():
                self.graphs[pdbid] = data_dict[pdbid]
               # graph = smiles_to_graph(smiles)
              #  if graph is not None:
              #      self.graphs[pdbid] = graph
               # else:
               #     print(f"Warning: Failed to create graph for {pdbid}")

        # Load sequence and pocket paths
        seq_path = data_path / phase / 'global'
        if seq_path.exists():
            self.seq_path = sorted(list(seq_path.glob('*.csv')))
        else:
            # Look for alternative sequence file format
            seq_df = pd.read_csv(data_path / f"{phase}_seq_.csv")
            self.seq_data = {row['id']: row['seq'] for _, row in seq_df.iterrows()}
            self.seq_path = list(self.seq_data.keys())
        self.max_seq_len = max_seq_len

        pkt_path = data_path / phase / 'pocket'
        if pkt_path.exists():
            self.pkt_path = sorted(list(pkt_path.glob('*.csv')))
        else:
            # Look for alternative pocket file format
            pkt_df = pd.read_csv(data_path / f"{phase}_pocket_.csv")
            self.pkt_data = {row['id']: row['seq'] for _, row in pkt_df.iterrows()}
            self.pkt_path = list(self.pkt_data.keys())
        self.max_pkt_len = max_pkt_len
        self.pkt_window = pkt_window
        self.pkt_stride = pkt_stride
        if self.pkt_window is None or self.pkt_stride is None:
            print(f'Dataset {phase}: will not fold pkt')

        # Get list of valid IDs (present in all required datasets)
        self.valid_ids = []
        for pdbid in self.smi.keys():
            if pdbid in self.affinity:
                # Check if sequence and pocket data is available
                if hasattr(self, 'seq_data'):
                    has_seq = pdbid in self.seq_data
                else:
                    has_seq = any(p.stem == pdbid for p in self.seq_path)
                    
                if hasattr(self, 'pkt_data'):
                    has_pkt = pdbid in self.pkt_data
                else:
                    has_pkt = any(p.stem == pdbid for p in self.pkt_path)
                
                # For GNN models, check graph availability
                if use_gnn:
                    has_graph = pdbid in self.graphs
                else:
                    has_graph = True
                    
                if has_seq and has_pkt and has_graph:
                    self.valid_ids.append(pdbid)
        
        # Create a mapping from valid index to pdbid
        self.idx_to_pdbid = {i: pdbid for i, pdbid in enumerate(self.valid_ids)}
        self.length = len(self.valid_ids)
        
        print(f"Dataset {phase}: loaded {self.length} valid samples")

    def __getitem__(self, idx: int) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
                                           Tuple[np.ndarray, np.ndarray, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray]]:
        """Get item by index."""
        pdbid = self.idx_to_pdbid[idx]
        
        # Load sequence tensor
        if hasattr(self, 'seq_data'):
            # Process sequence string data (needs implementation based on your encoding)
            # This is a placeholder - you'll need to replace with actual encoding logic
            _seq_tensor = np.zeros((min(len(self.seq_data[pdbid]), self.max_seq_len), PT_FEATURE_SIZE))
            # Implement sequence encoding here
        else:
            # Find the matching sequence file
            seq_file = next((p for p in self.seq_path if p.stem == pdbid), None)
            if seq_file:
                _seq_tensor = pd.read_csv(seq_file, index_col=0).drop(['idx'], axis=1).values[:self.max_seq_len]
            else:
                _seq_tensor = np.zeros((0, PT_FEATURE_SIZE))
        
        seq_tensor = np.zeros((self.max_seq_len, PT_FEATURE_SIZE))
        seq_tensor[:len(_seq_tensor)] = _seq_tensor

        # Load pocket tensor
        if hasattr(self, 'pkt_data'):
            # Process pocket string data (needs implementation based on your encoding)
            # This is a placeholder - you'll need to replace with actual encoding logic
            _pkt_tensor = np.zeros((min(len(self.pkt_data[pdbid]), self.max_pkt_len), PT_FEATURE_SIZE))
            # Implement pocket encoding here
        else:
            # Find the matching pocket file
            pkt_file = next((p for p in self.pkt_path if p.stem == pdbid), None)
            if pkt_file:
                _pkt_tensor = pd.read_csv(pkt_file, index_col=0).drop(['idx'], axis=1).values[:self.max_pkt_len]
            else:
                _pkt_tensor = np.zeros((0, PT_FEATURE_SIZE))
        
        if self.pkt_window is not None and self.pkt_stride is not None:
            pkt_len = (int(np.ceil((self.max_pkt_len - self.pkt_window) / self.pkt_stride))
                       * self.pkt_stride
                       + self.pkt_window)
            pkt_tensor = np.zeros((pkt_len, PT_FEATURE_SIZE))
            pkt_tensor[:len(_pkt_tensor)] = _pkt_tensor
            pkt_tensor = np.array(
                [pkt_tensor[i * self.pkt_stride:i * self.pkt_stride + self.pkt_window]
                 for i in range(int(np.ceil((self.max_pkt_len - self.pkt_window) / self.pkt_stride)))]
            )
        else:
            pkt_tensor = np.zeros((self.max_pkt_len, PT_FEATURE_SIZE))
            pkt_tensor[:len(_pkt_tensor)] = _pkt_tensor
        
        # Get ligand representation
        if self.use_gnn:
            # Return the graph directly - it will be batched by a custom collate function
            graph_data = self.graphs[pdbid]
            
            return (seq_tensor.astype(np.float32),
                    pkt_tensor.astype(np.float32),
                    graph_data,  # This will be a PyG Data object
                    np.array(self.affinity[pdbid], dtype=np.float32))
        else:
            # Use SMILES string representation
            return (seq_tensor.astype(np.float32),
                    pkt_tensor.astype(np.float32),
                    label_smiles(self.smi[pdbid], self.max_smi_len),
                    np.array(self.affinity[pdbid], dtype=np.float32))

    def __len__(self) -> int:
        """Return dataset length."""
        return self.length

def collate_gnn(batch):
    """
    Custom collate function for batching graph data.
    
    Args:
        batch: List of tuples (seq, pkt, graph, target)
        
    Returns:
        Tuple of batched tensors
    """
    # Separate the components
    seqs, pkts, graphs, targets = zip(*batch)
    
    # Convert to torch tensors where needed
    seq_tensor = torch.FloatTensor(np.stack(seqs))
    pkt_tensor = torch.FloatTensor(np.stack(pkts))
    
    # Batch the graphs
    batched_graph = Batch.from_data_list(graphs)
    graph_data = (
        batched_graph.x,
        batched_graph.edge_index,
        batched_graph.edge_attr,
        batched_graph.batch
    )
    
    # Convert targets to tensor
    target_tensor = torch.FloatTensor(np.stack(targets))
    
    return seq_tensor, pkt_tensor, graph_data, target_tensor