from dataclasses import dataclass
from pathlib import Path


@dataclass
class DTADataConfig:
    pic50_path: Path
    ligand_path: Path
    protein_path: Path
    output_path: Path
