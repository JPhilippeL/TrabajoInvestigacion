"""
@file ednn_model.py
@author Francesc Serratosa
@brief EDNN model definition.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, NNConv


class EDNNLayer(nn.Module):
    """
    @brief Simple edge-aware message-passing layer.
    @details
    This layer uses node features from source and target nodes plus one edge
    feature, usually the interatomic distance stored in data.edge_attr.
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 1):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return x

        row, col = edge_index

        m_ij = torch.cat([x[row], x[col], edge_attr], dim=1)
        m_ij = self.edge_mlp(m_ij)

        agg = torch.zeros_like(x)
        agg.index_add_(0, row, m_ij)

        update = self.node_mlp(torch.cat([x, agg], dim=1))
        return x + update


class EDNN(nn.Module):
    def __init__(self, node_dim=12, edge_dim=1, hidden_dim=64):
        super().__init__()

        nn_edge = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * node_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * node_dim, hidden_dim * node_dim),
        )

        self.conv1 = NNConv(node_dim, hidden_dim, nn_edge, aggr="mean")
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def _get_edge_attr(self, data):
        """
        @brief Return edge attributes with shape [num_edges, edge_dim].
        @details
        If data.edge_attr is missing, a fallback distance feature is computed
        from data.pos. If data.pos is missing too, a constant feature is used.
        """
        edge_index = data.edge_index
        num_edges = edge_index.shape[1]
        device = data.x.device

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_attr = data.edge_attr.float().to(device)
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            return edge_attr

        if hasattr(data, "pos") and data.pos is not None and num_edges > 0:
            row, col = edge_index
            dist = torch.norm(data.pos[row] - data.pos[col], dim=1, keepdim=True)
            return dist.float().to(device)

        return torch.ones((num_edges, self.edge_dim), dtype=torch.float, device=device)

    def forward(self, data):
        x, edge_index, edge_attr, batch = \
            data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x).view(-1)
