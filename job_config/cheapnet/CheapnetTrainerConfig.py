from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CheapnetTrainerConfig:
    output_path: Path
    train_split_file: Path
    val_split_file: Path
    test_split_file: Path
    graphs_path: Path
    batch_size: int
    lr: float
    hidden_dim: int
    weight_decay: float
    epochs: int
    drop_out: float
    patience: int
    node_dim: int = 14
    device = str = "cuda" if torch.cuda.is_available() else "cpu"
