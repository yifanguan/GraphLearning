import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops, to_torch_coo_tensor # add_self_loops might not be needed directly here
from torch_scatter import scatter_add, scatter # Used internally by MessagePassing with aggr='add'
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import BatchNorm
# from torch_geometric.nn import global_mean_pool, BatchNorm, global_add_pool

# from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

def deg_vec(edge_index):
    adj = to_dense_adj(edge_index)[0]
    deg = adj.sum(dim=1)
    return deg


class iMP(MessagePassing):
    """
    Injective message passing with D_{-1}A row scaling.
    """
    def __init__(self,
                 eps=2.0**0.5,
                 in_dim: int = -1,
                 out_dim: int = 300,
                 act=F.tanh,
                 freeze: bool = True):
        # Initialize the MessagePassing base class with 'add' aggregation
        super().__init__(aggr='add')
    
        self.eps = eps
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = act
        self.freeze = freeze

        self.W = nn.Linear(in_features=in_dim, out_features=out_dim)
        std = 1.0
        nn.init.normal_(self.W.weight, mean=0.0, std=std)

        # Train-free injective message passing, so freeze the parameters of weights
        if self.freeze:
            for param in self.W.parameters():
                param.requires_grad = False

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        row, col = edge_index
        deg = degree(col, num_nodes=num_nodes, dtype=x.dtype)

        h = self.W(x)
        h = self.act(h)
        h = h / deg.unsqueeze(1)
        # h = h / self.out_dim**0.5
        Ah = torch.zeros_like(h).index_add(0, row, h[col])

        return Ah



class DecoupleModel(nn.Module):
    """
    A Graph Neural Network model.

    Args:
        num_mp_layers: The number of injective layers to stack.
        num_fl_layers: number of feature learning layers to stack.
        eps (float): The epsilon value to use for all layers.
        act (nn.Module, optional): The activation function to apply between
                                         layers (e.g., nn.ReLU()). If None, no activation
                                         is applied. Defaults to None.
    """
    def __init__(self,
                in_dim: int,
                mp_width: int,
                fl_width: int,
                out_dim:int,
                num_mp_layers: int = 3,
                num_fl_layers: int = 2,
                eps = 2.0**0.5,
                act=F.relu,
                freeze=True,
                dropout: float = 0
                # batch_normalization: bool = False,
                # skip_connection: bool = True,
                #first_layer_linear: bool = True
                ):
        super().__init__()
        self.act = act
        self.dropout = dropout
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            iMP(in_dim=in_dim if i == 0 else mp_width, out_dim=mp_width, eps=eps, act=act, freeze=freeze)
            for i in range(num_mp_layers)
        ]) if num_mp_layers > 0 else None

        # Fully connected layers
        self.fc_layers = nn.ModuleList(
            [nn.Linear(mp_width if self.mp_layers else in_dim, fl_width)] +
            [nn.Linear(fl_width, fl_width) for _ in range(num_fl_layers - 1)]
        ) if num_fl_layers > 0 else None

        # Per-layer injection projections from MP output to FC width
        self.injection_projs = nn.ModuleList([
            nn.Linear(mp_width if self.mp_layers else in_dim, fl_width)
            for _ in range(num_fl_layers)
        ]) if num_fl_layers > 0 else None

        # Output layer
        out_input_dim = (
            fl_width if self.fc_layers else
            mp_width if self.mp_layers else
            in_dim
        )
        self.output_layer = nn.Linear(out_input_dim, out_dim)

    def forward(self, data):
        """
        Forward pass through the stacked GNN layers.
        Returns:
            torch.Tensor: Node features after passing through all layers.
        """
        x, edge_index = data.x, data.edge_index

        if self.mp_layers:
            for layer in self.mp_layers:
                x = layer(x, edge_index)

        if self.fc_layers:
            x_inject = x  # Save for injection
            for layer, proj in zip(self.fc_layers, self.injection_projs):
                injected = proj(x_inject)
                x = self.alpha * layer(self.act(x)) + injected
                x = F.dropout(x, p=self.dropout, training=self.training)

        return self.output_layer(x)


class MPOnlyModel(nn.Module):
    """
    A truncated version of our decouple model used with only stacked message passing layers
    """
    def __init__(self, model):
        super().__init__()
        self.mp_layers = model.mp_layers

    def forward(self, h, edge_index, index):
        """
        Forward pass through the stacked GNN layers.
        Returns:
            torch.Tensor: Node features after passing through all layers.
        """
        h = self.mp_layers[index](h, edge_index)

        return h


