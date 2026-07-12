from pathlib import Path

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from data_pipeline.atom_utils import ATOM_TYPES_DTA, one_of_k_encoding, one_of_k_encoding_unk
from data_pipeline.ligand_utils import ligand_graph_dta
from data_pipeline.pic50_utils import load_pic50
from data_pipeline.protein_utils import load_protein_sequence, seq_cat
from job_config.graphdta.DTADataConfig import DTADataConfig


class DTAGraph:
    def __init__(self, config: DTADataConfig):
        self.config = config

    def _build_dta_data(self, pdb_id, pic50):
        sdf_path = self.config.ligand_path / f"{pdb_id}_ligand.sdf"
        pdb_path = self.config.protein_path / f"{pdb_id}_protein.pdb"

        if not Path.is_file(sdf_path):
            print(f"[ERROR GRAPHDTA LIGAND FILE]: Missing file {sdf_path}")
            return None

        if not Path.is_file(pdb_path):
            print(f"[ERROR GRAPHDTA PROTEIN FILE]: Missing file {pdb_path}")
            return None

        ligand_graph = ligand_graph_dta(sdf_path)
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
            y=torch.tensor([float(pic50)], dtype=torch.float32),
        )
        data.target = torch.tensor(target, dtype=torch.long).unsqueeze(0)
        data.pdb_id = pdb_id

        return data

    def generate_all_dta_dta(self):
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        pic50_dict = load_pic50(self.config.pic50_path)
        for pdb_id, pic50 in tqdm(pic50_dict.items(), desc="GRAPH DTA data generation"):
            data = self._build_dta_data(pdb_id, pic50)

            if data is None:
                continue
            out_path = self.config.output_path / f"{pdb_id}.pt"
            torch.save(data, out_path)
