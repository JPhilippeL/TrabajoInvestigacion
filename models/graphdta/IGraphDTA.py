from abc import ABC, abstractmethod

import torch.nn as nn


class GraphDTA(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, data):
        pass
