from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CheapnetPredictionConfig:
    pic50_path: Path
    test_split_file: Path
    graphs_path: Path
    model_path: Path
    output_path: Path
    batch_size: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
