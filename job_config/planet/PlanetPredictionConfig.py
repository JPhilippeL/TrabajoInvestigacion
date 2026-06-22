from dataclasses import dataclass
from pathlib import Path
from typing import Type

import torch
import torch.nn as nn


@dataclass
class PlanetPredictionConfig:
    data_output_path: Path

    model_path: Path

    output_path: Path

    model: Type[nn.Module]

    model_name: str

    batch_size: int

    num_workers: int

    seed: int
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.data_output_path = Path(self.data_output_path)
        self.model_path = Path(self.model_path)
        self.output_path = Path(self.output_path)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
