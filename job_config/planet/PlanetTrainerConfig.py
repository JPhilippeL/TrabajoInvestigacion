import typing
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class PlanetTrainerConfig:
    output_path: Path

    data_output_path: Path

    model: typing.Type[nn.Module]

    model_name: str

    batch_size: int

    epochs: int

    lr: float

    weight_decay: float

    patience: int

    seed: int

    num_workers: int

    feature_dims: int

    nheads: int

    key_dims: int

    value_dims: int

    pro_update_inters: int

    lig_update_iters: int

    pro_lig_update_iters: int

    clip_norm: float

    beta_start_step: int

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
