from dataclasses import dataclass, field
from pathlib import Path

import torch

from models.graphdta.IGraphDTA import GraphDTA


@dataclass
class DTAPredictionConfig:
    model: type[GraphDTA]
    model_name: str

    pic50_path: Path
    model_path: Path
    graphs_path: Path
    test_split_file: Path
    output_path: Path

    batch_size: int = 32

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
