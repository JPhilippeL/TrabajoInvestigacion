import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any
import URVDEEPTAF.Core.urvdtaf_metrics as urvdtaf_metrics
from torch_geometric.nn import NNConv, global_add_pool
from .urvdtaf_dataset import PT_FEATURE_SIZE, GNN_NODE_FEATURE_SIZE

CHAR_SMI_SET_LEN = 64

class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Squeeze tensor to remove singleton dimensions."""
        return input.squeeze()

class CDilated(nn.Module):
    def __init__(self, nIn: int, nOut: int, kSize: int, stride: int = 1, d: int = 1):
        """
        Convolutional dilated layer.
        
        Args:
            nIn: Number of input channels
            nOut: Number of output channels
            kSize: Kernel size
            stride: Stride
            d: Dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        output = self.conv(input)
        return output

class DilatedParllelResidualBlockA(nn.Module):
    def __init__(self, nIn: int, nOut: int, add: bool = True):
        """
        Dilated Parallel Residual Block Type A.
        
        Args:
            nIn: Number of input channels
            nOut: Number of output channels
            add: Whether to add residual connection
        """
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16)  # dilation rate of 2^4
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
            # print(f'{nIn}-{nOut}: add=False')
            add = False
        self.add = add

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # reduce
        output1 = self.c1(input)
        output1 = self.br1(output1)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # hierarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        # merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output

class DilatedParllelResidualBlockB(nn.Module):
    def __init__(self, nIn: int, nOut: int, add: bool = True):
        """
        Dilated Parallel Residual Block Type B.
        
        Args:
            nIn: Number of input channels
            nOut: Number of output channels
            add: Whether to add residual connection
        """
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
            # print(f'{nIn}-{nOut}: add=False')
            add = False
        self.add = add

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # reduce
        output1 = self.c1(input)
        output1 = self.br1(output1)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        # hierarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        # merge
        combine = torch.cat([d1, add1, add2, add3], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output

class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for molecular representation.
    Builds its NNConv layers eagerly in __init__, based on a known edge_feat_dim.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        edge_feat_dim: int = 6  # 4 bond types + conjugation + ring membership
    ):
        super().__init__()
        self.node_in       = input_dim
        self.hidden        = hidden_dim
        self.out           = output_dim
        self.edge_feat_dim = edge_feat_dim

        # 1) Edge MLP for conv1
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(edge_feat_dim, input_dim * hidden_dim),
            nn.ReLU(),
            nn.Linear(input_dim * hidden_dim, input_dim * hidden_dim),
        )
        self.conv1 = NNConv(input_dim, hidden_dim, self.edge_mlp1, aggr='mean')

        # 2) Edge MLP for conv2
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim * hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim),
        )
        self.conv2 = NNConv(hidden_dim, hidden_dim, self.edge_mlp2, aggr='mean')

        # 3) Edge MLP for conv3
        self.edge_mlp3 = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim * output_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * output_dim, hidden_dim * output_dim),
        )
        self.conv3 = NNConv(hidden_dim, output_dim, self.edge_mlp3, aggr='mean')

        # BatchNorm + activation
        self.bn    = nn.BatchNorm1d(output_dim)
        self.prelu = nn.PReLU()

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_attr)

        # Readout: sum over nodes in each graph
        x = global_add_pool(x, batch)  # [batch_size, output_dim]
        x = self.bn(x)
        x = self.prelu(x)
        return x


    def _build_convs(self, edge_feat_dim):
        """Build the NNConv layers and their edge‐MLP once we know edge_feat_dim."""
        self.edge_feat_dim = edge_feat_dim

        # MLP for conv1: maps edge → (node_in * hidden)
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(edge_feat_dim, self.node_in * self.hidden),
            nn.ReLU(),
            nn.Linear(self.node_in * self.hidden, self.node_in * self.hidden),
        )
        self.conv1 = NNConv(self.node_in, self.hidden, self.edge_mlp1, aggr='mean')

        # MLP for conv2: maps edge → (hidden * hidden)
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(edge_feat_dim, self.hidden * self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden * self.hidden, self.hidden * self.hidden),
        )
        self.conv2 = NNConv(self.hidden, self.hidden, self.edge_mlp2, aggr='mean')

        # MLP for conv3: maps edge → (hidden * out)
        self.edge_mlp3 = nn.Sequential(
            nn.Linear(edge_feat_dim, self.hidden * self.out),
            nn.ReLU(),
            nn.Linear(self.hidden * self.out, self.hidden * self.out),
        )
        self.conv3 = NNConv(self.hidden, self.out, self.edge_mlp3, aggr='mean')

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass through GNN encoder.
        
        Args:
            x: Node features (num_nodes, num_features)
            edge_index: Edge indices (2, num_edges)
            batch: Batch indices (num_nodes)
            
        Returns:
            Graph embedding (batch_size, output_dim)
        """
        # Node embeddings
        # on first pass, build NNConv layers
        if self.edge_feat_dim is None:
            # edge_attr.shape = [num_edges, edge_feat_dim]
            self._build_convs(edge_attr.size(1))

        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        
        # Readout - aggregate node features to graph representation
        x = global_add_pool(x, batch)  # [batch_size, output_dim]
        
        # Normalization and activation
        x = self.bn(x)
        x = self.prelu(x)
        
        return x

class BaseDeepDTAF(nn.Module):
    """
    Base class for all DeepDTAF model variants.
    """
    def __init__(self, 
                 use_protein: bool = True, 
                 use_pocket: bool = True,
                 use_ligand: bool = True,
                 use_gnn: bool = False,
                 seq_embed_size: int = 128,
                 smi_embed_size: int = 128,
                 seq_oc: int = 128,
                 pkt_oc: int = 128,
                 smi_oc: int = 128,
                 gnn_node_feat: int = GNN_NODE_FEATURE_SIZE,
                 gnn_hidden: int = 64):
        """
        Initialize base DeepDTAF model.
        
        Args:
            use_protein: Whether to use global protein features
            use_pocket: Whether to use pocket features
            use_ligand: Whether to use ligand features
            use_gnn: Whether to use GNN for ligand representation
            seq_embed_size: Sequence embedding size
            smi_embed_size: SMILES embedding size
            seq_oc: Output channels for sequence
            pkt_oc: Output channels for pocket
            smi_oc: Output channels for SMILES
            gnn_node_feat: Number of node features for GNN
            gnn_hidden: Hidden dimension for GNN
        """
        super().__init__()
        
        self.use_protein = use_protein
        self.use_pocket = use_pocket
        self.use_ligand = use_ligand
        self.use_gnn = use_gnn
        
        # Check that at least one feature type is used
        if not (use_protein or use_pocket or use_ligand):
            raise ValueError("At least one of protein, pocket, or ligand features must be used")
        
        # --- CORRECCIÓN EN LIGANDO ---
        if use_ligand:
            if use_gnn:
                self.gnn_encoder = GNNEncoder(
                    input_dim=gnn_node_feat,
                    hidden_dim=gnn_hidden,
                    output_dim=smi_oc,
                    edge_feat_dim=6
                )
            else:
                self.smi_embed = nn.Embedding(CHAR_SMI_SET_LEN, smi_embed_size)
                conv_smi = []
                ic = smi_embed_size
                for oc in [32, 64, smi_oc]:
                    conv_smi.append(DilatedParllelResidualBlockB(ic, oc))
                    ic = oc
                conv_smi.append(nn.AdaptiveMaxPool1d(1))
                # --- BORRADO: conv_smi.append(Squeeze()) --- <--- ELIMINAR ESTA LÍNEA
                self.conv_smi = nn.Sequential(*conv_smi)

        # Shared embedding for sequence and pocket
        if use_protein or use_pocket:
            self.seq_embed = nn.Linear(PT_FEATURE_SIZE, seq_embed_size)
        
        # --- CORRECCIÓN EN SECUENCIA ---
        if use_protein:
            conv_seq = []
            ic = seq_embed_size
            for oc in [32, 64, 64, seq_oc]:
                conv_seq.append(DilatedParllelResidualBlockA(ic, oc))
                ic = oc
            conv_seq.append(nn.AdaptiveMaxPool1d(1))
            # --- BORRADO: conv_seq.append(Squeeze()) --- <--- ELIMINAR ESTA LÍNEA
            self.conv_seq = nn.Sequential(*conv_seq)
        
        # --- CORRECCIÓN EN BOLSILLO ---
        if use_pocket:
            conv_pkt = []
            ic = seq_embed_size
            for oc in [32, 64, pkt_oc]:
                conv_pkt.append(nn.Conv1d(ic, oc, 3))
                conv_pkt.append(nn.BatchNorm1d(oc))
                conv_pkt.append(nn.PReLU())
                ic = oc
            conv_pkt.append(nn.AdaptiveMaxPool1d(1))
            # --- BORRADO: conv_pkt.append(Squeeze()) --- <--- ELIMINAR ESTA LÍNEA
            self.conv_pkt = nn.Sequential(*conv_pkt)
        
        # Calculate combined feature size
        combined_size = 0
        if use_protein:
            combined_size += seq_oc
        if use_pocket:
            combined_size += pkt_oc
        if use_ligand:
            combined_size += smi_oc
        
        self.cat_dropout = nn.Dropout(0.2)
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64, 1),
            nn.PReLU())
    
    def forward(self, seq: Optional[torch.Tensor] = None, 
                pkt: Optional[torch.Tensor] = None, 
                smi: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], List]] = None) -> torch.Tensor:
        """
        Forward pass through the model using explicit flattening.
        """
        features = []
        
        # -----------------------------------------------------------
        # 1. PROCESAR SECUENCIA
        # -----------------------------------------------------------
        if self.use_protein:
            if seq is None:
                raise ValueError("Sequence tensor is required when use_protein=True")
            
            seq_embed = self.seq_embed(seq)            # [Batch, Len, 128]
            seq_embed = torch.transpose(seq_embed, 1, 2) # [Batch, 128, Len]
            seq_conv = self.conv_seq(seq_embed)        # [Batch, 128, 1] (Tras AdaptivePool)
            
            # CAMBIO CRÍTICO: Aplanar explícitamente a [Batch, 128]
            seq_conv = seq_conv.view(seq_conv.size(0), -1) 
            
            features.append(seq_conv)
        
        # -----------------------------------------------------------
        # 2. PROCESAR BOLSILLO
        # -----------------------------------------------------------
        if self.use_pocket:
            if pkt is None:
                raise ValueError("Pocket tensor is required when use_pocket=True")
                
            pkt_embed = self.seq_embed(pkt)
            pkt_embed = torch.transpose(pkt_embed, 1, 2)
            pkt_conv = self.conv_pkt(pkt_embed)        # [Batch, 128, 1]
            
            # CAMBIO CRÍTICO: Aplanar explícitamente a [Batch, 128]
            pkt_conv = pkt_conv.view(pkt_conv.size(0), -1)
            
            features.append(pkt_conv)
        
        # -----------------------------------------------------------
        # 3. PROCESAR LIGANDO
        # -----------------------------------------------------------
        if self.use_ligand:
            if smi is None:
                raise ValueError("SMILES tensor or graph data is required when use_ligand=True")
                
            if self.use_gnn:
                # GNN suele devolver ya [Batch, 128], pero aseguramos
                if isinstance(smi, (tuple, list)) and len(smi) == 4:
                    x, edge_index, edge_attr, batch = smi
                    gnn_out = self.gnn_encoder(x, edge_index, edge_attr, batch)
                    features.append(gnn_out)
                else:
                    raise ValueError(f"GNN requires tuple/list of 4 elements, got {type(smi)}")
            else:
                # SMILES Convencional
                smi_embed = self.smi_embed(smi)
                smi_embed = torch.transpose(smi_embed, 1, 2)
                smi_conv = self.conv_smi(smi_embed)    # [Batch, 128, 1]
                
                # CAMBIO CRÍTICO: Aplanar explícitamente a [Batch, 128]
                smi_conv = smi_conv.view(smi_conv.size(0), -1)
                
                features.append(smi_conv)
        
        # -----------------------------------------------------------
        # 4. CONCATENACIÓN (Sin parches)
        # -----------------------------------------------------------
        # Ahora features tiene [Tensor(B, 128), Tensor(B, 128), Tensor(B, 128)]
        # Al concatenar en dim=1, obtenemos [Batch, 384]
        
        cat = torch.cat(features, dim=1)
        cat = self.cat_dropout(cat)
        
        # Final classification
        output = self.classifier(cat)
        
        return output

# Model A: Original DeepDTAF with all components
class DeepDTAF(BaseDeepDTAF):
    """
    DeepDTAF model (Model A) with all components:
    - Global protein features
    - Local pocket features
    - Ligand features
    """
    def __init__(self):
        """Initialize DeepDTAF model."""
        super().__init__(use_protein=True, use_pocket=True, use_ligand=True, use_gnn=False)

# Model B: DeepDTAF without pocket
class DeepDTAF_NoPocket(BaseDeepDTAF):
    """
    DeepDTAF model with pocket removed (Model B):
    - Global protein features
    - Ligand features
    """
    def __init__(self):
        """Initialize DeepDTAF model without pocket."""
        super().__init__(use_protein=True, use_pocket=False, use_ligand=True, use_gnn=False)
    
    def forward(self, seq: torch.Tensor, pkt: Optional[torch.Tensor], smi: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            seq: Sequence tensor (N, L, 40)
            pkt: Pocket tensor (N, L, 40) - ignored in this model
            smi: SMILES tensor (N, L)
            
        Returns:
            Predicted affinity values (N, 1)
        """
        return super().forward(seq=seq, smi=smi)

# Model C: DeepDTAF without protein
class DeepDTAF_NoProtein(BaseDeepDTAF):
    """
    DeepDTAF model with protein removed (Model C):
    - Local pocket features
    - Ligand features
    """
    def __init__(self):
        """Initialize DeepDTAF model without protein."""
        super().__init__(use_protein=False, use_pocket=True, use_ligand=True, use_gnn=False)
    
    def forward(self, seq: Optional[torch.Tensor], pkt: torch.Tensor, smi: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            seq: Sequence tensor (N, L, 40) - ignored in this model
            pkt: Pocket tensor (N, L, 40)
            smi: SMILES tensor (N, L)
            
        Returns:
            Predicted affinity values (N, 1)
        """
        return super().forward(pkt=pkt, smi=smi)

# Model D: DeepDTAF with only ligand
class DeepDTAF_OnlyLigand(BaseDeepDTAF):
    """
    DeepDTAF model with only ligand features (Model D):
    - Ligand features
    """
    def __init__(self):
        """Initialize DeepDTAF model with only ligand."""
        super().__init__(use_protein=False, use_pocket=False, use_ligand=True, use_gnn=False)
    
    def forward(self, seq: Optional[torch.Tensor], pkt: Optional[torch.Tensor], smi: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            seq: Sequence tensor (N, L, 40) - ignored in this model
            pkt: Pocket tensor (N, L, 40) - ignored in this model
            smi: SMILES tensor (N, L)
            
        Returns:
            Predicted affinity values (N, 1)
        """
        return super().forward(smi=smi)

# Model E: DeepDTAF with GNN for ligand
class DeepDTAF_GNN(BaseDeepDTAF):
    """
    DeepDTAF model with GNN for ligand (Model E):
    - Global protein features
    - Local pocket features
    - Ligand features with GNN
    """
    def __init__(self):
        """Initialize DeepDTAF model with GNN for ligand."""
        super().__init__(use_protein=True, use_pocket=True, use_ligand=True, use_gnn=True)
    
    def forward(self, seq: torch.Tensor, pkt: torch.Tensor, 
                graph_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            seq: Sequence tensor (N, L, 40)
            pkt: Pocket tensor (N, L, 40)
            graph_data: Tuple of (x, edge_index, batch) for GNN
            
        Returns:
            Predicted affinity values (N, 1)
        """
        return super().forward(seq=seq, pkt=pkt, smi=graph_data)

# Model F: DeepDTAF with GNN for ligand, without pocket
class DeepDTAF_GNN_NoPocket(BaseDeepDTAF):
    """
    DeepDTAF model with GNN for ligand, without pocket (Model F):
    - Global protein features
    - Ligand features with GNN
    """
    def __init__(self):
        """Initialize DeepDTAF model with GNN for ligand, without pocket."""
        super().__init__(use_protein=True, use_pocket=False, use_ligand=True, use_gnn=True)
    
    def forward(self, seq: torch.Tensor, pkt: Optional[torch.Tensor], 
                graph_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            seq: Sequence tensor (N, L, 40)
            pkt: Pocket tensor (N, L, 40) - ignored in this model
            graph_data: Tuple of (x, edge_index, batch) for GNN
            
        Returns:
            Predicted affinity values (N, 1)
        """
        return super().forward(seq=seq, smi=graph_data)

# Model G: DeepDTAF with GNN for ligand, without protein
class DeepDTAF_GNN_NoProtein(BaseDeepDTAF):
    """
    DeepDTAF model with GNN for ligand, without protein (Model G):
    - Local pocket features
    - Ligand features with GNN
    """
    def __init__(self):
        """Initialize DeepDTAF model with GNN for ligand, without protein."""
        super().__init__(use_protein=False, use_pocket=True, use_ligand=True, use_gnn=True)
    
    def forward(self, seq: Optional[torch.Tensor], pkt: torch.Tensor, 
                graph_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            seq: Sequence tensor (N, L, 40) - ignored in this model
            pkt: Pocket tensor (N, L, 40)
            graph_data: Tuple of (x, edge_index, batch) for GNN
            
        Returns:
            Predicted affinity values (N, 1)
        """
        return super().forward(pkt=pkt, smi=graph_data)

# Model H: DeepDTAF with GNN for ligand, without protein and pocket
class DeepDTAF_GNN_OnlyLigand(BaseDeepDTAF):
    """
    DeepDTAF model with GNN for ligand, without protein and pocket (Model H):
    - Ligand features with GNN
    """
    def __init__(self):
        """Initialize DeepDTAF model with GNN for ligand, without protein and pocket."""
        super().__init__(use_protein=False, use_pocket=False, use_ligand=True, use_gnn=True)
    
    def forward(self, seq: Optional[torch.Tensor], pkt: Optional[torch.Tensor], 
                graph_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            seq: Sequence tensor (N, L, 40) - ignored in this model
            pkt: Pocket tensor (N, L, 40) - ignored in this model
            graph_data: Tuple of (x, edge_index, batch) for GNN
            
        Returns:
            Predicted affinity values (N, 1)
        """
        return super().forward(smi=graph_data)

def test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    show: bool = True
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate `model` on `test_loader`, returning metrics plus raw predictions.

    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test split
        loss_function: Loss function (e.g., nn.MSELoss)
        device: torch.device ("cpu" or "cuda")
        show: Whether to display a tqdm progress bar

    Returns:
        A tuple of:
          - metrics: dict with keys 'loss', 'RMSE', 'MAE', 'CORR', 'MSE', 'R2'
          - outputs: np.ndarray of shape (N,) with model predictions
          - targets: np.ndarray of shape (N,) with ground-truth values
    """
    model.eval()
    total_loss = 0.0
    outputs_list, targets_list = [], []

    with torch.no_grad():
        for batch_data in tqdm(test_loader, disable=not show, total=len(test_loader)):
            # Separate the target from inputs
            *x, y = batch_data
            
            # Process inputs based on model type
            processed_x = []
            for i, xi in enumerate(x):
                if isinstance(xi, (list, tuple)):
                    # Handle GNN graph data or other structured data
                    if len(xi) == 4:  # Assume it's (x, edge_index, edge_attr, batch)
                        # Convert list to tuple and move each tensor to device
                        graph_tuple = tuple(
                            t.to(device) if torch.is_tensor(t) else t 
                            for t in xi
                        )
                        processed_x.append(graph_tuple)
                    else:
                        # Handle other list/tuple structures
                        processed_data = []
                        for item in xi:
                            if torch.is_tensor(item):
                                processed_data.append(item.to(device))
                            else:
                                processed_data.append(item)
                        processed_x.append(tuple(processed_data) if isinstance(xi, tuple) else processed_data)
                elif torch.is_tensor(xi):
                    # Handle regular tensors
                    processed_x.append(xi.to(device))
                else:
                    # Handle other types
                    processed_x.append(xi)
            
            y = y.to(device)

            # Forward pass
            y_hat = model(*processed_x)

            # Accumulate loss
            total_loss += loss_function(y_hat.view(-1), y.view(-1)).item()

            # 1. CAMBIO AQUÍ: Añadimos .detach() por seguridad absoluta
            outputs_list.append(y_hat.detach().cpu().numpy().reshape(-1))
            targets_list.append(y.detach().cpu().numpy().reshape(-1))

    # 2. CAMBIO AQUÍ: Forzamos tipo float64 (el que le gusta a Numba)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)
    targets = np.concatenate(targets_list, axis=0).astype(np.float64)

    # Concatenate batches
    outputs = np.concatenate(outputs_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    # Normalize loss by dataset size
    avg_loss = total_loss / len(test_loader.dataset)

    # Compute additional metrics
    metrics_dict = {
        'loss': avg_loss,
        'RMSE': urvdtaf_metrics.RMSE(targets, outputs),
        'MAE': urvdtaf_metrics.MAE(targets, outputs),
        'CORR': urvdtaf_metrics.CORR(targets, outputs),
        'MSE': urvdtaf_metrics.MSE(targets, outputs),
        'R2': urvdtaf_metrics.R2(targets, outputs),
    }

    return metrics_dict, outputs, targets


# Dictionary mapping model names to model classes
MODEL_DICT = {
    'DeepDTAF': DeepDTAF,                  # Model A
    'DeepDTAF_NoPocket': DeepDTAF_NoPocket,          # Model B
    'DeepDTAF_NoProtein': DeepDTAF_NoProtein,        # Model C
    'DeepDTAF_OnlyLigand': DeepDTAF_OnlyLigand,      # Model D
    'DeepDTAF_GNN': DeepDTAF_GNN,              # Model E
    'DeepDTAF_GNN_NoPocket': DeepDTAF_GNN_NoPocket,     # Model F
    'DeepDTAF_GNN_NoProtein': DeepDTAF_GNN_NoProtein,    # Model G
    'DeepDTAF_GNN_OnlyLigand': DeepDTAF_GNN_OnlyLigand,   # Model H
}

# Dictionary indicating which models use GNN
GNN_MODELS = {
    'DeepDTAF_GNN': True,
    'DeepDTAF_GNN_NoPocket': True,
    'DeepDTAF_GNN_NoProtein': True,
    'DeepDTAF_GNN_OnlyLigand': True,
    'DeepDTAF': False,
    'DeepDTAF_NoPocket': False,
    'DeepDTAF_NoProtein': False,
    'DeepDTAF_OnlyLigand': False,
}

MODEL_MAPPINGS = {
    # Letter mappings (Model A -> DeepDTAF)
    'Model A': 'DeepDTAF',
    'Model B': 'DeepDTAF_NoPocket',
    'Model C': 'DeepDTAF_NoProtein',
    'Model D': 'DeepDTAF_OnlyLigand',
    'Model E': 'DeepDTAF_GNN',
    'Model F': 'DeepDTAF_GNN_NoPocket',
    'Model G': 'DeepDTAF_GNN_NoProtein',
    'Model H': 'DeepDTAF_GNN_OnlyLigand',
    
    # Allow no space variants (ModelA -> DeepDTAF)
    'ModelA': 'DeepDTAF',
    'ModelB': 'DeepDTAF_NoPocket',
    'ModelC': 'DeepDTAF_NoProtein',
    'ModelD': 'DeepDTAF_OnlyLigand',
    'ModelE': 'DeepDTAF_GNN',
    'ModelF': 'DeepDTAF_GNN_NoPocket',
    'ModelG': 'DeepDTAF_GNN_NoProtein',
    'ModelH': 'DeepDTAF_GNN_OnlyLigand',

    # Model Name Mapping
    'DeepDTAF': DeepDTAF,                  # Model A
    'DeepDTAF_NoPocket': DeepDTAF_NoPocket,          # Model B
    'DeepDTAF_NoProtein': DeepDTAF_NoProtein,        # Model C
    'DeepDTAF_OnlyLigand': DeepDTAF_OnlyLigand,      # Model D
    'DeepDTAF_GNN': DeepDTAF_GNN,              # Model E
    'DeepDTAF_GNN_NoPocket': DeepDTAF_GNN_NoPocket,     # Model F
    'DeepDTAF_GNN_NoProtein': DeepDTAF_GNN_NoProtein,    # Model G
    'DeepDTAF_GNN_OnlyLigand': DeepDTAF_GNN_OnlyLigand,   # Model H

    # Model Name Mapping Spaces
    'DeepDTAF': DeepDTAF,                  # Model A
    'DeepDTAF No Pocket': DeepDTAF_NoPocket,          # Model B
    'DeepDTAF No Protein': DeepDTAF_NoProtein,        # Model C
    'DeepDTAF Only Ligand': DeepDTAF_OnlyLigand,      # Model D
    'DeepDTAF GNN': DeepDTAF_GNN,              # Model E
    'DeepDTAF GNN No Pocket': DeepDTAF_GNN_NoPocket,     # Model F
    'DeepDTAF GNN No Protein': DeepDTAF_GNN_NoProtein,    # Model G
    'DeepDTAF GNN Only Ligand': DeepDTAF_GNN_OnlyLigand,   # Model H

     # Model Name Mapping No _
    'DeepDTAF': DeepDTAF,                  # Model A
    'DeepDTAFNoPocket': DeepDTAF_NoPocket,          # Model B
    'DeepDTAFNoProtein': DeepDTAF_NoProtein,        # Model C
    'DeepDTAFOnlyLigand': DeepDTAF_OnlyLigand,      # Model D
    'DeepDTAFGNN': DeepDTAF_GNN,              # Model E
    'DeepDTAFGNNNoPocket': DeepDTAF_GNN_NoPocket,     # Model F
    'DeepDTAFGNNNoProtein': DeepDTAF_GNN_NoProtein,    # Model G
    'DeepDTAFGNNOnlyLigand': DeepDTAF_GNN_OnlyLigand,   # Model H

    'deepdtaf': DeepDTAF,                  # Model A
    'deepdtafnopocket': DeepDTAF_NoPocket,          # Model B
    'deepdtafnoprotein': DeepDTAF_NoProtein,        # Model C
    'deepdtafonlyligand': DeepDTAF_OnlyLigand,      # Model D
    'deepdtafgnn': DeepDTAF_GNN,              # Model E
    'deepdtafgnnnopocket': DeepDTAF_GNN_NoPocket,     # Model F
    'deepdtafgnnnoprotein': DeepDTAF_GNN_NoProtein,    # Model G
    'deepdtafgnnonlyligand': DeepDTAF_GNN_OnlyLigand,   # Model H

    'DEEPDTAF': DeepDTAF,                  # Model A
    'DEEPDTAFNOPOCKET': DeepDTAF_NoPocket,          # Model B
    'DEEPDTAFNOPROTEIN': DeepDTAF_NoProtein,        # Model C
    'DEEPDTAFONLYLIGAND': DeepDTAF_OnlyLigand,      # Model D
    'DEEPDTAFGNN': DeepDTAF_GNN,              # Model E
    'DEEPDTAFGNNNOPOCKET': DeepDTAF_GNN_NoPocket,     # Model F
    'DEEPDTAFGNNNOPROTEIN': DeepDTAF_GNN_NoProtein,    # Model G
    'DEEPDTAFGNNONLYLIGAND': DeepDTAF_GNN_OnlyLigand,   # Model H

    'deepdtaf': DeepDTAF,                  # Model A
    'deepdtaf no pocket': DeepDTAF_NoPocket,          # Model B
    'deepdtaf no protein': DeepDTAF_NoProtein,        # Model C
    'deepdtaf only ligand': DeepDTAF_OnlyLigand,      # Model D
    'deepdtaf gnn': DeepDTAF_GNN,              # Model E
    'deepdtaf gnn no pocket': DeepDTAF_GNN_NoPocket,     # Model F
    'deepdtaf gnn no protein': DeepDTAF_GNN_NoProtein,    # Model G
    'deepdtaf gnn only ligand': DeepDTAF_GNN_OnlyLigand,   # Model H

    'DEEPDTAF': DeepDTAF,                  # Model A
    'DEEPDTAF NO POCKET': DeepDTAF_NoPocket,          # Model B
    'DEEPDTAF NO PROTEIN': DeepDTAF_NoProtein,        # Model C
    'DEEPDTAF ONLY LIGAND': DeepDTAF_OnlyLigand,      # Model D
    'DEEPDTAF GNN': DeepDTAF_GNN,              # Model E
    'DEEPDTAF GNN NO POCKET': DeepDTAF_GNN_NoPocket,     # Model F
    'DEEPDTAF GNN NO PROTEIN': DeepDTAF_GNN_NoProtein,    # Model G
    'DEEPDTAF GNN ONLY LIGAND': DeepDTAF_GNN_OnlyLigand,   # Model H
}

