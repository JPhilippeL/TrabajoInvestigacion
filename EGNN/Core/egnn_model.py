"""
@file egnn_model.py
@author Francesc Serratosa
@brief EGNN model definition.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class EGNNLayer(nn.Module):
    """
    @brief Basic equivariant-style message passing layer.

    The layer uses node features and squared inter-node distances to build
    messages. Coordinates are used to compute distances but are not updated.
    """

    def __init__(self, feat_dim: int):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2 + 1, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x

        row, col = edge_index

        diff = pos[row] - pos[col]
        dist2 = (diff ** 2).sum(dim=1, keepdim=True)

        messages = torch.cat([x[row], x[col], dist2], dim=1)
        messages = self.edge_mlp(messages)

        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, row, messages)

        return x + self.node_mlp(aggregated)


class EGNN(nn.Module):
    """
    @brief EGNN regression model for protein-ligand graphs.
    """

    def __init__(self, node_dim: int = 12, hidden_dim: int = 64):
        super().__init__()

        self.embed = nn.Linear(node_dim, hidden_dim)

        self.conv1 = EGNNLayer(hidden_dim)
        self.conv2 = EGNNLayer(hidden_dim)
        self.conv3 = EGNNLayer(hidden_dim)

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data) -> torch.Tensor:
        x = data.x
        pos = data.pos
        edge_index = data.edge_index
        batch = data.batch

        x = self.embed(x)

        x = F.relu(self.conv1(x, pos, edge_index))
        x = F.relu(self.conv2(x, pos, edge_index))
        x = F.relu(self.conv3(x, pos, edge_index))

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))

        return self.lin2(x).view(-1)
