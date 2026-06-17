import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from models.gign.cheapnet_v2 import MLP, GIGNBlock


class GIGN(nn.Module):
    def __init__(self, node_dim, hidden_dim, drop_rate):
        super().__init__()

        # Embedding
        self.embedding = MLP(node_dim, hidden_dim, 0.0)

        # GIGN blocks (intra + inter)
        self.block1 = GIGNBlock(hidden_dim, hidden_dim, drop_rate)
        self.block2 = GIGNBlock(hidden_dim, hidden_dim, drop_rate)
        self.block3 = GIGNBlock(hidden_dim, hidden_dim, drop_rate)

        # Final predictor
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        # Embedding
        x = self.embedding(data.x)

        # GIGN blocks (requiere edge_index_intra e edge_index_inter)
        x = self.block1(x, data)
        x = self.block2(x, data)
        x = self.block3(x, data)

        # Pooling global por grafo
        x = global_mean_pool(x, data.batch)

        # FC
        x = self.fc(x)

        return x.view(-1)