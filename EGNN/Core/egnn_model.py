import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class EGNNLayer(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2 + 1, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU()
        )

    def forward(self, x, pos, edge_index):

        row, col = edge_index

        # distancia cuadrada
        diff = pos[row] - pos[col]
        dist2 = (diff ** 2).sum(dim=1, keepdim=True)

        # mensaje
        m_ij = torch.cat([x[row], x[col], dist2], dim=1)
        m_ij = self.edge_mlp(m_ij)

        # agregación
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, m_ij)

        # actualización nodo
        x = x + self.node_mlp(agg)

        return x


class EGNN(nn.Module):
    def __init__(self, node_dim=12, hidden_dim=64):
        super().__init__()

        self.embed = nn.Linear(node_dim, hidden_dim)

        self.conv1 = EGNNLayer(hidden_dim)
        self.conv2 = EGNNLayer(hidden_dim)
        self.conv3 = EGNNLayer(hidden_dim)

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):

        x, pos, edge_index, batch = \
            data.x, data.pos, data.edge_index, data.batch

        x = self.embed(x)

        x = F.relu(self.conv1(x, pos, edge_index))
        x = F.relu(self.conv2(x, pos, edge_index))
        x = F.relu(self.conv3(x, pos, edge_index))

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))

        return self.lin2(x).view(-1)
