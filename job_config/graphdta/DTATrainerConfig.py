from dataclasses import dataclass
from pathlib import Path

import torch

from models.graphdta.IGraphDTA import GraphDTA


@dataclass
class DTATrainerConfig:
    model: type[GraphDTA]
    model_name: str

    output_path: Path
    graphs_path: Path

    train_split_file: Path
    val_split_file: Path
    test_split_file: Path

    dropout: float
    number_of_filters: int
    batch_size: int
    lr: float
    weight_decay: float
    epochs: int

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    patience: int = 15
