from models.graphdta.architecture.GAT import GATNet
from models.graphdta.architecture.GAT_GCN import GAT_GCN
from models.graphdta.architecture.GCN import GCNNet
from models.graphdta.architecture.GINConv import GINConvNet

GRAPH_DTA_MODELS = {
    "GCNNet": GCNNet,
    "GATNet": GATNet,
    "GAT_GCN": GAT_GCN,
    "GINConvNet": GINConvNet,
}


def get_graph_dta_model(model_name: str):
    if model_name not in GRAPH_DTA_MODELS:
        raise ValueError(f"Unknown GraphDTA model: {model_name}")

    return GRAPH_DTA_MODELS[model_name]
