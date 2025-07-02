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

class InjectiveMP(MessagePassing):
    """
    An injective layer with normalization to prevent large magnitude.
    I is the identity matrix, H is the node feature matrix, and epsilon is a irrational scalar used to distinguish node itself from its neighbors.
    This layer itself has no learnable parameters. (Train-Free, ideally)
    No skip connection, no batch normalization, no dropout should be used
    We assume all MPs have the same width m
    """
    def __init__(self,
                 eps=2.0**0.5,
                 in_dim: int = -1,
                 out_dim: int = 300,
                 act=F.relu,
                 freeze: bool = True):
        # Initialize the MessagePassing base class with 'add' aggregation
        super().__init__(aggr='add')
    
        self.eps = eps
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = act
        self.freeze = freeze

        self.W = nn.Linear(in_features=in_dim, out_features=out_dim)
        # set std of W
        # self.adj_matrix = to_dense_adj(edge_index).squeeze(0) 
        # norm = self.spectral_norm(edge_index)
        # std = (1.0 / (1 + epsilon + norm))
        std = 1.0
        nn.init.normal_(self.W.weight, mean=0.0, std=std)

        # Train-free injective message passing, so freeze the parameters of weights
        if self.freeze:
            for param in self.W.parameters():
                param.requires_grad = False

    # def turn_off_training(self):
    #     if self.linear:
    #         for param in self.linear.parameters():
    #             param.requires_grad = False
            # for param in self.batch_norm.parameters():
            #     param.requires_grad = False

    # def turn_on_training(self):
    #     if self.linear:
    #         for param in self.linear.parameters():
    #             param.requires_grad = True
            # for param in self.batch_norm.parameters():
            #     param.requires_grad = True

    # def lift_operation(self, h):
    #     h = self.linear(h)
    #     h = F.relu(h)
    #     h = h * torch.tensor(self.m).pow(-0.5)

    #     return h

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        target, source = edge_index
        deg = degree(source, num_nodes=num_nodes, dtype=x.dtype)
        norm_factor = 1 + self.eps + deg
        # norm = avg_degree(edge_index)
        # norm = max_degree(edge_index)

        h = self.W(x)
        h = self.act(h)
        h = h / self.out_dim**0.5
        h = h / norm_factor.unsqueeze(1)
        Ah = torch.zeros_like(h).index_add(0, target, h[source])
        
        return (1 + self.eps) * h + Ah


class DecoupleModel(nn.Module):
    """
    A Graph Neural Network model stacking multiple CustomGNNLayer layers.

    Args:
        num_layers (int): The number of InjectiveGNNLayer layers to stack.
        num_linear_layers: number of feature learning layers
        epsilon (float): The epsilon value to use for all layers.
        activation (nn.Module, optional): The activation function to apply between
                                         layers (e.g., nn.ReLU()). If None, no activation
                                         is applied. Defaults to None.
        first_layer_linear: if the first layer mapping from input_dim to hidden dim, a nn.linear layer
                            or our injective layer
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
                freeze=True
                # dropout: float = 0.2,
                # batch_normalization: bool = False,
                # skip_connection: bool = True,
                #first_layer_linear: bool = True
                ):
        super().__init__()
        self.act = act

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            InjectiveMP(in_dim=in_dim if i == 0 else mp_width, out_dim=mp_width, eps=eps, act=act, freeze=freeze)
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

        Args:
            x (torch.Tensor): Initial node features [num_nodes, num_features].
            edge_index (torch.Tensor): Graph connectivity [2, num_edges].
            edge_weight (torch.Tensor, optional): Edge weights [num_edges]. Defaults to None.

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
                x = layer(self.act(x)) + injected

        return self.output_layer(x)



# def deg_vec(edge_index):
#     adj = to_dense_adj(edge_index)[0]
#     deg = adj.sum(dim=1)
#     return deg

# class iMP(nn.Module):
#     def __init__(self, in_dim, out_dim, eps=2.0**0.5, 
#                  act=F.relu, freeze=False):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.eps = eps
#         self.act = act
#         self.freeze = freeze

#         self.W = nn.Linear(in_dim, out_dim, bias=False)
#         torch.nn.init.normal_(self.W.weight, mean=0.0, std=1.0)

#         if self.freeze:
#             for param in self.W.parameters():
#                 param.requires_grad = False  # freeze weights

#     def forward(self, x, edge_index):
#         degree = deg_vec(edge_index)
#         norm_factor = 1 + self.eps + degree
#         # norm = avg_degree(edge_index)
#         # norm = max_degree(edge_index)
        
#         h = self.W(x)
#         h = self.act(h)
#         h = h / self.out_dim**0.5
#         h = h / norm_factor.unsqueeze(1)

#         row, col = edge_index
#         Ah = torch.zeros_like(h).index_add(0, row, h[col])
#         return (1 + self.eps) * h + Ah

# class iGNN(nn.Module):
#     def __init__(self, in_dim, mp_width, fc_width, out_dim, 
#                  num_mp_layers=3, num_fc_layers=2, 
#                  eps=torch.sqrt(torch.tensor(2.0)), act=F.gelu, freeze=False):
#         super().__init__()
#         self.act = act

#         # Message passing layers (optional)
#         self.mp_layers = nn.ModuleList([
#             iMP(in_dim if i == 0 else mp_width, mp_width, eps=eps, act=act, freeze=freeze)
#             for i in range(num_mp_layers)
#         ]) if num_mp_layers > 0 else None

#         # Fully connected layers (optional)
#         self.fc_layers = nn.ModuleList(
#             [nn.Linear(mp_width if self.mp_layers else in_dim, fc_width)] +
#             [nn.Linear(fc_width, fc_width) for _ in range(num_fc_layers - 1)]
#         ) if num_fc_layers > 0 else None

#         # Per-layer injection projections from MP output to FC width
#         self.injection_projs = nn.ModuleList([
#             nn.Linear(mp_width if self.mp_layers else in_dim, fc_width)
#             for _ in range(num_fc_layers)
#         ]) if num_fc_layers > 0 else None

#         # Output layer
#         out_input_dim = (
#             fc_width if self.fc_layers else
#             mp_width if self.mp_layers else
#             in_dim
#         )
#         self.output_layer = nn.Linear(out_input_dim, out_dim)


#     def forward(self, data: Data):
#         x, edge_index = data.x, data.edge_index

#         if self.mp_layers:
#             for layer in self.mp_layers:
#                 x = layer(x, edge_index)

#         # FC layers with per-layer injection
#         if self.fc_layers:
#             x_inject = x  # Save for injection
#             for layer, proj in zip(self.fc_layers, self.injection_projs):
#                 injected = proj(x_inject)
#                 x = layer(self.act(x)) + injected

#         return self.output_layer(x)