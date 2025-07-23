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
                 in_dim: int = -1,
                 out_dim: int = 300,
                 act=F.gelu,
                 freeze: bool = True,
                 alpha = 1.0,
                 skip_connection=False,
                 simple=False):
        # Initialize the MessagePassing base class with 'add' aggregation
        super().__init__(aggr='add')
    
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = act
        self.freeze = freeze
        self.skip_connection = skip_connection
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float))
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.simple = simple

        # Train-free injective message passing, so freeze the parameters of weights
        if self.freeze:
            for param in self.linear.parameters():
                param.requires_grad = False
            self.alpha.requires_grad = False

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        # includes self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        # use in degree; in case of undirected graph, this is the same as out degree
        deg = degree(col, num_nodes=num_nodes, dtype=x.dtype)

        h = x
        # Compute T @ act(x) @ W
        if not self.simple:
            h = self.act(x)
        h = h / deg.unsqueeze(1) # safe now, no divide by 0 issue
        h = torch.zeros_like(h).index_add(0, col, h[row])
        if not self.simple:
            h = self.linear(h)

        # pre-activation skip connection
        if self.skip_connection and self.linear.in_features == self.linear.out_features:
            return x + self.alpha * h
        return h


class iGNN(nn.Module):
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
                act=F.gelu,
                freeze=True,
                dropout: float = 0,
                alpha=1.0,
                skip_connection = False
                ):
        super().__init__()
        self.act = act
        self.dropout = dropout
        self.skip_connection = skip_connection
        self.alpha = alpha

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            iMP(in_dim=in_dim if i == 0 else mp_width, out_dim=mp_width, act=act, freeze=freeze, alpha=alpha, skip_connection=skip_connection)
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
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.fc_layers:
            x_inject = x  # Save for injection
            for layer, proj in zip(self.fc_layers, self.injection_projs):
                injected = proj(x_inject)
                x = layer(self.act(x)) + injected
                # x = F.dropout(x, p=self.dropout, training=self.training)

        return self.output_layer(x)




class iGNN_energy_version(nn.Module):
    """
    A Graph Neural Network model.
    Train with energy as part of the loss.

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
                out_dim:int,
                num_mp_layers: int = 3,
                act=F.gelu,
                freeze=True,
                # dropout: float = 0,
                alpha=1.0,
                skip_connection = False
                ):
        super().__init__()
        self.act = act
        # self.dropout = dropout
        self.skip_connection = skip_connection
        self.alpha = alpha

        self.input_layer = nn.Linear(in_dim, mp_width)

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            iMP(in_dim=mp_width, out_dim=mp_width, act=act, freeze=freeze, alpha=alpha, skip_connection=skip_connection)
            for i in range(num_mp_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(mp_width, out_dim)

    def forward(self, data):
        """
        Forward pass through the stacked GNN layers.
        Returns:
            torch.Tensor: Node features after passing through all layers.
        """
        x, edge_index = data.x, data.edge_index

        h = self.input_layer(x)

        for layer in self.mp_layers:
            h = layer(h, edge_index)

        return self.output_layer(h)





class iGNN_V2(nn.Module):
    """
    A Graph Neural Network model.
    Each mp is followed by a fc.

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
                act=F.gelu,
                freeze=True,
                dropout: float = 0,
                alpha=1.0,
                skip_connection = False
                ):
        # In V2, number of message passing layer is equal to number fully connected layers
        assert num_mp_layers == num_fl_layers
        assert mp_width == fl_width
        super().__init__()
        self.act = act
        self.dropout = dropout
        self.skip_connection = skip_connection
        self.alpha = alpha

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            iMP(in_dim=in_dim if i == 0 else mp_width, out_dim=mp_width, act=act, freeze=freeze, alpha=alpha, skip_connection=skip_connection)
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

        # if self.mp_layers:
        #     for layer in self.mp_layers:
        #         x = layer(x, edge_index)

        # if self.fc_layers:
        #     x_inject = x  # Save for injection
        #     for layer, proj in zip(self.fc_layers, self.injection_projs):
        #         injected = proj(x_inject)
        #         x = layer(self.act(x)) + injected
        #         x = F.dropout(x, p=self.dropout, training=self.training)

        for i, layer in enumerate(self.mp_layers):
            x = layer(x, edge_index)
            x_inject = x  # Save for injection
            injected = self.injection_projs[i](x_inject)
            x = self.fc_layers[i](self.act(x)) + injected
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.output_layer(x)



# class InjectiveMP(MessagePassing):
#     """
#     An injective layer with normalization to prevent large magnitude.
#     I is the identity matrix, H is the node feature matrix, and epsilon is a irrational scalar used to distinguish node itself from its neighbors.
#     This layer itself has no learnable parameters. (Train-Free, ideally)
#     No skip connection, no batch normalization, no dropout should be used
#     We assume all MPs have the same width m
#     """
#     def __init__(self,
#                  eps=2.0**0.5,
#                  in_dim: int = -1,
#                  out_dim: int = 300,
#                  act=F.tanh,
#                  freeze: bool = True):
#         # Initialize the MessagePassing base class with 'add' aggregation
#         super().__init__(aggr='add')
    
#         self.eps = eps
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.act = act
#         self.freeze = freeze

#         self.W = nn.Linear(in_features=in_dim, out_features=out_dim)
#         # set std of W
#         # self.adj_matrix = to_dense_adj(edge_index)'.squeeze(0) '
#         # norm = self.spectral_norm(edge_index)
#         # std = (1.0 / (1 + epsilon + norm))
#         std = 1.0
#         nn.init.normal_(self.W.weight, mean=0.0, std=std)

#         # Train-free injective message passing, so freeze the parameters of weights
#         if self.freeze:
#             for param in self.W.parameters():
#                 param.requires_grad = False

#     # def turn_off_training(self):
#     #     if self.linear:
#     #         for param in self.linear.parameters():
#     #             param.requires_grad = False
#             # for param in self.batch_norm.parameters():
#             #     param.requires_grad = False

#     # def turn_on_training(self):
#     #     if self.linear:
#     #         for param in self.linear.parameters():
#     #             param.requires_grad = True
#             # for param in self.batch_norm.parameters():
#             #     param.requires_grad = True

#     # def lift_operation(self, h):
#     #     h = self.linear(h)
#     #     h = F.relu(h)
#     #     h = h * torch.tensor(self.m).pow(-0.5)

#     #     return h

#     def forward(self, x, edge_index):
#         num_nodes = x.size(0)
#         row, col = edge_index
#         deg = degree(col, num_nodes=num_nodes, dtype=x.dtype)
#         norm_factor = 1 + self.eps + deg
#         # norm = avg_degree(edge_index)
#         # norm = max_degree(edge_index)

#         h = self.W(x)
#         h = self.act(h)
#         h = h / self.out_dim**0.5
#         h = h / norm_factor.unsqueeze(1)
#         Ah = torch.zeros_like(h).index_add(0, row, h[col])

#         return (1 + self.eps) * h + Ah


class InjectiveMP(MessagePassing):
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
        # set std of W
        # self.adj_matrix = to_dense_adj(edge_index)'.squeeze(0) '
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
        row, col = edge_index
        deg = degree(col, num_nodes=num_nodes, dtype=x.dtype)
        # norm_factor = 1 + self.eps + deg
        # norm = avg_degree(edge_index)
        # norm = max_degree(edge_index)

        h = self.W(x)
        h = self.act(h)
        h = h / deg.unsqueeze(1)
        # h = h / self.out_dim**0.5
        # h = h / norm_factor.unsqueeze(1)
        Ah = torch.zeros_like(h).index_add(0, row, h[col])

        return Ah
        # return (1 + self.eps) * h + Ah



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
        Returns:
            torch.Tensor: Node features after passing through all layers.
        """
        x, edge_index = data.x, data.edge_index

        if self.mp_layers:
            for layer in self.mp_layers:
                x = layer(x, edge_index)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.fc_layers:
            x_inject = x  # Save for injection
            for layer, proj in zip(self.fc_layers, self.injection_projs):
                injected = proj(x_inject)
                x = layer(self.act(x)) + injected
                # x = F.dropout(x, p=self.dropout, training=self.training)

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


