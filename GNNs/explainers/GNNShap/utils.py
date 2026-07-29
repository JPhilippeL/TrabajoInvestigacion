
import logging
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data import Data
import math
import torch.nn.functional as F

def get_coalition_counts(mask_matrix: Union[np.array, Tensor]) -> np.array:
    """Finds counts of each coalition size for a given mask matrix.

    Args:
        mask_matrix (Union[np.array, Tensor]): mask matrix obtained from a sampler

    Returns:
        np.array: coalition counts
    """
    if torch.is_tensor(mask_matrix):
        mask_matrix = mask_matrix.cpu().numpy()
    coal_sizes = mask_matrix.sum(1).astype(int)
    unique, counts = np.unique(coal_sizes, return_counts=True)
    return counts


def get_coalition_size_weights(mask_matrix: Union[np.array, Tensor],
                               weights: Union[np.array, Tensor]) -> np.array:
    """Finds sum of total weights for each coalition size.

    Args:
        mask_matrix (Union[np.array, Tensor]): mask matrix obtained from a sampler
        weights (Union[np.array, Tensor]): weights vector obtained from a sampler

    Returns:
        np.array: coalition size weights
    """
    if torch.is_tensor(mask_matrix):
        mask_matrix = mask_matrix.cpu().numpy()

    if torch.is_tensor(weights):
        weights = weights.cpu().numpy()

    counts = mask_matrix.sum(1)
    nplayers = mask_matrix.shape[1]

    weight_sums = np.zeros(nplayers -1)
    for i in range(1, nplayers):
        weight_sums[i-1] = weights[counts == i].sum()
    return weight_sums


def get_gnn_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Finds and returns GNN layers.

    Args:
        model (torch.nn.Module): pyg model.

    Returns:
        List[torch.nn.Module]: GNN layers as a list
    """
    gnn_layers = []
    for module in model.modules():
        if isinstance(module, MessagePassing):
            gnn_layers.append(module)
    return gnn_layers

def switch_add_self_loops(model: torch.nn.Module):
    """Switches each layers add_self_loops value to True or False.

    Args:
        model (torch.nn.Module): pyg model.
    """
    layers = get_gnn_layers(model)
    for layer in layers:
        layer.add_self_loops = not layer.add_self_loops

def switch_normalize(model: torch.nn.Module):
    """Switches each layers normalize value to True or False.

    Args:
        model (torch.nn.Module): pyg model.
    """
    layers = get_gnn_layers(model)
    for layer in layers:
        layer.normalize = not layer.normalize

def has_normalization(model: torch.nn.Module) -> bool:
    """Checks if gnn layers have normalization. It controls whether all layers
    have same configuration.

    Args:
        model (torch.nn.Module): pyg model.

    Raises:
        AssertionError: Raises assertion error if different layers have different configurations.
        AssertionError: Raises assertion error if there is no gnn layers.

    Returns:
        bool: boolean value whether gnn layers have normalization
    """
    layers = get_gnn_layers(model)
    if len(layers) > 0:
        try: # some GNN types has no normalize attribute
            normalize = layers[0].normalize
        except:
            return False
        if len(layers) > 1:
            for layer in layers[1:]:
                if layer.normalize != normalize:
                    raise AssertionError(("Layers have different normalization settings."
                                         " This is not supported!"))
        return normalize
    raise AssertionError("No GNN layers found!")


def has_add_self_loops(model: torch.nn.Module) -> bool:
    """Checks if model adds self loops. It controls whether all layers have same configuration.

    Args:
        model (torch.nn.Module): pyg model.

    Raises:
        AssertionError: Raises assertion error if different layers have different configurations.
        AssertionError: Raises assertion error if there is no gnn layers.

    Returns:
        bool: boolean value whether model adds self loops or not.
    """

    layers = get_gnn_layers(model)
    if len(layers) > 0:
        try:
            self_loop = layers[0].add_self_loops
        except:
            return False

        if len(layers) > 1:
            for layer in layers[1:]:
                if layer.add_self_loops != self_loop:
                    raise AssertionError(("Layers have different add_self_loops settings."
                                         " This is not supported!"))
        return self_loop
    raise AssertionError("No GNN layers found!")


@torch.no_grad()
def pruned_comp_graph(node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
    directed: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Finds the pruned computational graph for a given node index. Similar to k_hop_subgraph, but
    k_hop_subgraph returns all edges between k-hop nodes. We are only interested in edges that
    carries message to target node in k_hops."""

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    my_edge_mask = row.new_empty(row.size(0), dtype=torch.bool) # added by sakkas
    my_edge_mask.fill_(False) # added by sakkas

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)# input, dimension, index
        my_edge_mask[edge_mask] = True
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    edge_index = edge_index[:, my_edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, my_edge_mask

def node2edge_score(edge_index: torch.Tensor, node_scores: np.array):
    """Converts node scores to edge scores: an edge score is equal to average of connected nodes. 
    Needed for some baselines that only provide node scores.

    Args:
        edge_index (torch.Tensor): PyG edge index.
        node_scores (np.array[float]): node scores

    Returns:
        np.array: edge scores
    """

    edge_scores = np.zeros(edge_index.size(1))
    np_node_scores = np.array(node_scores)
    edge_scores += np_node_scores[edge_index[0].cpu().numpy()]
    edge_scores += np_node_scores[edge_index[1].cpu().numpy()]
    return edge_scores/2



def fidelity(node_data: dict, data: Data, model: torch.nn.Module, sparsity: float = 0.3,
               fid_type: str = 'neg', topk: int = 0, target_class: int = None,
               apply_abs: bool=True) -> tuple:
    """Computes fidelity+ and fidelity- score of a node. It supports both topk and sparsity. 
    If sparsity set to 0.3, it drops 30% of the edges. Based on the neg or pos, it drops 
    unimportant or important edges. It applies topk based keep if topk is set to a positive 
    integer other than zero.

    Note that it computes fidelity scores for the predicted class if target class is not provided.

    Args:
        node_data (dict): a node's explanation data with node_id, num_players, scores keys.
        data (Data): pyG Data.
        model (torch.nn.Module): a PyTorch model.
        sparsity (float, optional): target sparsity value. Defaults to 0.3.
        fid_type (str, optional): Fidelity type: neg or pos. Defaults to 'neg'.
        topk (int, optional): Topk edges to keep. Defaults to 0.
        target_class (int, optional): Target class to compute fidelity score. Defaults to None.
        apply_abs (bool, optional): applies absolute to scores. Some methods can find negative and 
            positive contributing nodes/edges. Fidelity-wise, we only care the change amount. We can 
            use this to get rid of negative contributing edges to improve accuracy. Defaults to 
            True.

    Returns:
        tuple: node_id, nplayers, fidelity score, current sparsity, correct_class, init_pred_class, 
            and sparse_pred_class.
    """
    assert topk >= 0, "topk cannot be a negative number"
    assert 0 <= sparsity <= 1, "Sparsity should be between zero and one."

    node_id = int(node_data['node_id'])
    correct_class = data.y[node_id].item()

    model.eval()

    

    # find khop computational graph
    (subset, sub_edge_index, new_node_id,
     _) = pruned_comp_graph(node_id, model.num_layers, data.edge_index, relabel_nodes=True)
    # new node id due to relabeling
    new_node_id = int(new_node_id[0].cpu().numpy())
    num_initial_edges = sub_edge_index.size(1)  # number of players


    subset = subset.cpu().numpy()

    # initial prediction
    init_pred = F.softmax(model(data.x[subset], sub_edge_index), dim=1)[new_node_id]
    init_pred_class = init_pred.argmax(dim=-1).item()
    if target_class is None:
        target_class = init_pred_class
    init_prob = init_pred[target_class].item()


    if node_data['num_players'] == num_initial_edges:
        edge_scores = np.array(node_data['scores'])

    # convert node scores to edge scores if node score is provided
    elif node_data['num_players'] == subset.shape[0]:
        edge_scores = node2edge_score(sub_edge_index, node_data['scores'])

    else:
        raise ValueError("Number of players should be equal to either"
                        " number of edges or number of nodes!")


    edge_scores = np.abs(edge_scores) if apply_abs else edge_scores


    # less important edge at first index
    edge_importance_sorted = edge_scores.argsort()

    if topk == 0:  # sparsity based
        if fid_type == 'pos':  # reverse the list: most important edge at first index
            edge_importance_sorted = edge_scores.argsort()[::-1].copy()
            # copy required for bug fixing. pytorch doesn't support negative index

        # how many edges to drop
        drop_len = num_initial_edges - math.ceil(num_initial_edges * (1 - sparsity))
        keep_edges = edge_importance_sorted[drop_len:]

    else:  # topk based
        if fid_type == 'neg':
            keep_edges = edge_importance_sorted[topk:]  # drop least important topk edges
        else:  # fid+
            keep_edges = edge_importance_sorted[:-topk] # keep edges except topk

        drop_len = num_initial_edges - len(keep_edges)


    keep_edges.sort()

    sparse_pred = F.softmax(model(data.x[subset], sub_edge_index[:, keep_edges]),
                            dim=-1)[new_node_id]
    
    sparse_pred_class = sparse_pred.argmax(dim=-1).item()
    sparse_prob = sparse_pred[target_class].item()

    prob_score = sparse_prob - init_prob
    prob_score = np.abs(prob_score) if apply_abs else prob_score


    current_sparsity = drop_len / num_initial_edges
    return (node_id, num_initial_edges, prob_score, current_sparsity,
            correct_class, init_pred_class, sparse_pred_class)