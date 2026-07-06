"""
@file model.py
@author Mohamed EL BOUKHIARI
@brief Dynamic WideDTA CNN model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WideCNN(nn.Module):
    """
    @brief WideDTA CNN with dynamic input dimensions.
    @details
    The original code hardcoded Davis/KIBA dimensions in Conv1d and Linear layers.
    This implementation uses LazyConv1d and LazyLinear so that Davis, KIBA and MPro
    input shapes can be initialized from the first batch.
    """

    def __init__(
        self,
        conv1_channels: int = 16,
        conv2_channels: int = 32,
        fc_hidden_dim: int = 512,
        fc_middle_dim: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.pconv1 = nn.LazyConv1d(conv1_channels, kernel_size=2, stride=1, padding=1)
        self.pconv2 = nn.Conv1d(conv1_channels, conv2_channels, kernel_size=2, stride=1, padding=1)

        self.lconv1 = nn.LazyConv1d(conv1_channels, kernel_size=2, stride=1, padding=1)
        self.lconv2 = nn.Conv1d(conv1_channels, conv2_channels, kernel_size=2, stride=1, padding=1)

        self.mconv1 = nn.LazyConv1d(conv1_channels, kernel_size=2, stride=1, padding=1)
        self.mconv2 = nn.Conv1d(conv1_channels, conv2_channels, kernel_size=2, stride=1, padding=1)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

        self.FC1 = nn.LazyLinear(fc_hidden_dim)
        self.FC2 = nn.Linear(fc_hidden_dim, fc_middle_dim)
        self.FC3 = nn.Linear(fc_middle_dim, 1)

    def _encode_branch(self, x: torch.Tensor, conv1: nn.Module, conv2: nn.Module) -> torch.Tensor:
        x = F.relu(conv1(x))
        x = self.maxpool(x)
        x = F.relu(conv2(x))
        x = self.maxpool(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        """
        @brief Forward pass.
        @param x1 Protein tensor shaped [batch, protein_channels, protein_length].
        @param x2 Ligand tensor shaped [batch, ligand_channels, ligand_length].
        @param x3 Motif tensor shaped [batch, motif_channels, motif_length].
        """
        protein_features = self._encode_branch(x1, self.pconv1, self.pconv2)
        ligand_features = self._encode_branch(x2, self.lconv1, self.lconv2)
        motif_features = self._encode_branch(x3, self.mconv1, self.mconv2)

        x = torch.cat((protein_features, ligand_features, motif_features), dim=1)
        x = F.relu(self.FC1(x))
        x = self.dropout(x)
        x = F.relu(self.FC2(x))
        return self.FC3(x)
