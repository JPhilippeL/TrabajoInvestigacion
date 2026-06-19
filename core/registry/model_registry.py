from graphdta.architecture.GAT import GATNet
from graphdta.architecture.GAT_GCN import GAT_GCN
from graphdta.architecture.GCN import GCNNet
from graphdta.architecture.GINConv import GINConvNet

from facade_pattern.cheapnet_facade import CheapNetFacade
from facade_pattern.gign_facade import GIGNFacade
from facade_pattern.graph_dta_facade import DTAFacade

MODEL_REGISTRY = {
    "cheapnet": {
        "display_name": "CheapNet",
        "facade": CheapNetFacade(),
        "tasks": {
            "generate_data": "Generate dataset",
            "train": "Train model",
            "predict": "Predict",
        },
    },
    "gign": {
        "display_name": "GIGN",
        "facade": GIGNFacade(),
        "tasks": {
            "generate_data": "Generate dataset",
            "train": "Train model",
            "predict": "Predict",
        },
    },
    "graphdta": {
        "display_name": "GraphDTA",
        "facade": DTAFacade(),
        "tasks": {
            "generate_data": "Generate dataset",
            "train": "Train model",
            "predict": "Predict",
        },
    },
}
