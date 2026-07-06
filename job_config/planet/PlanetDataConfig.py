from dataclasses import dataclass
from pathlib import Path


@dataclass
class PlanetDataConfig:
    protein_path: Path
    ligand_path: Path
    pic50_path: Path
    output_path: Path

    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    seed: int = 42

    def __post_init__(self):
        self.protein_path = Path(self.protein_path)
        self.ligand_path = Path(self.ligand_path)
        self.pic50_path = Path(self.pic50_path)
        self.output_path = Path(self.output_path)

        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")

        if not 0 < self.valid_ratio < 1:
            raise ValueError("valid_ratio must be between 0 and 1")

        if self.train_ratio + self.valid_ratio >= 1:
            raise ValueError("train_ratio + valid_ratio must be < 1")
