from pathlib import Path
from time import time

import torch
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
from tqdm import tqdm

from data_pipeline.ligand_utils import load_ligand
from data_pipeline.pic50_utils import load_pic50
from data_pipeline.protein_utils import (
    filter_protein_atoms_by_distance,
    load_protein,
)
from job_config.cheapnet.CheapnetDataConfig import CheapnetDataConfig


class CheapnetDataGeneration:
    def __init__(self, config: CheapnetDataConfig):
        self.config = config
        self.pic50_dict = load_pic50(self.config.pic50_path)

    def _build_graph(self, pdb_id):
        ligand_file = Path(self.config.ligand_path) / f"{pdb_id}_ligand.sdf"
        protein_file = Path(self.config.protein_path) / f"{pdb_id}_protein.pdb"
        lig_coords, lig_feats, lig_mol = load_ligand(ligand_file)
        prot_coords, prot_feats, _ = load_protein(protein_file)

        if not lig_coords or not prot_coords or lig_mol is None:
            return None
        if pdb_id not in self.pic50_dict:
            return None

        prot_coords = filter_protein_atoms_by_distance(
            prot_coords, lig_coords, cutoff=self.config.cutoff_prot
        )
        prot_feats = {k: prot_feats[k] for k in prot_coords}

        if not prot_coords:
            return None

        pos_l_tensor = torch.stack([lig_coords[k] for k in lig_coords])
        pos_p_tensor = torch.stack([prot_coords[k] for k in prot_coords])

        pos = torch.cat([pos_l_tensor, pos_p_tensor], dim=0)

        feat_list = [lig_feats[k] for k in lig_feats] + [prot_feats[k] for k in prot_feats]
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
                if i != j and dist_pp[i, j] < self.config.dis_threshold:
                    edges_pro.append([i + num_l, j + num_l])

        edge_index_intra_pro = (
            torch.tensor(edges_pro, dtype=torch.long).T
            if edges_pro
            else torch.empty(2, 0, dtype=torch.long)
        )

        edge_index_intra = torch.cat([edge_index_intra_lig, edge_index_intra_pro], dim=1)

        dist_lp = distance_matrix(pos_l_tensor.numpy(), pos_p_tensor.numpy())

        edges_inter = []
        for i in range(num_l):
            for j in range(num_p):
                if dist_lp[i, j] < self.config.dis_threshold:
                    edges_inter.append([i, j + num_l])
                    edges_inter.append([j + num_l, i])

        edge_index_inter = (
            torch.tensor(edges_inter, dtype=torch.long).T
            if edges_inter
            else torch.empty(2, 0, dtype=torch.long)
        )

        edge_index_total = torch.cat([edge_index_intra, edge_index_inter], dim=1)

        deg = torch.bincount(edge_index_total[0], minlength=pos.shape[0]).float().unsqueeze(1)

        dist_feat = torch.zeros(pos.shape[0], 1)
        for i, j in edge_index_total.t():
            dist_feat[i] += torch.norm(pos[i] - pos[j])

        dist_feat /= torch.clamp(deg, min=1)

        x = torch.cat([x, deg, dist_feat], dim=1)

        y = torch.tensor([self.pic50_dict[pdb_id]], dtype=torch.float32)

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

    def build_graphs_pt(self, log_callback=None):
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        start_time = time()
        generated = 0
        for pdb_id in tqdm(self.pic50_dict.keys(), desc="Graph Generation CheapNet"):
            g = self._build_graph(pdb_id)
            if g is None:
                continue
            output_file = output_path / f"{pdb_id}.pt"
            torch.save(g, output_file)
            generated += 1
        end_time = time()
        message = (
            f"Cheapnet graph generation completed in {end_time - start_time:.2f} seconds."
            f" Generated : {generated}"
        )
        if log_callback:
            log_callback(message)

        return {
            "generated": generated,
            "duration": end_time - start_time,
            "output_path": str(self.config.output_path),
        }
