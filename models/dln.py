import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops, to_torch_coo_tensor # add_self_loops might not be needed directly here
from torch_scatter import scatter_add # Used internally by MessagePassing with aggr='add'
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import BatchNorm


# from torch_geometric.nn import global_mean_pool, BatchNorm, global_add_pool

# from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class InjectiveGNNLayer(MessagePassing):
    """
    An injective layer with normalization to prevent large magnitude.
    I is the identity matrix, H is the node feature matrix, and epsilon is a irrational scalar used to distinguish node itself from its neighbors.
    This layer itself has no learnable parameters. (Train-Free, ideally)
    No skip connection, no batch normalization, no dropout should be used
    We assume all MPs have the same width m
    """
    def __init__(self, edge_index: torch.Tensor, epsilon: float, input_dim: int = -1,
                 hidden_dim: int = 300, lift: bool = True, lift_first=True):
        """
        Args:
            epsilon (float): The epsilon value in the layer formula.
            linear: if this layer models a pure linear message passing. When not pure linear, a lift operation is added.
                    a lift operation itself is useful to make multi-layer InjectiveGNNLayer overall becomes injective. A lift operation is modeled as 
                    a MLP.
            lift_first: if the layer do lift first and then do message passing
        """
        # Initialize the MessagePassing base class with 'add' aggregation
        super().__init__(aggr='add')

        self.epsilon = float(epsilon) # Store epsilon as a float
        self.linear = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        # set std of W
        self.adj_matrix = to_dense_adj(edge_index).squeeze(0) 
        norm = self.spectral_norm(self.adj_matrix)
        std = 1.0 / (1 + epsilon + norm) 
        nn.init.normal_(self.linear.weight, mean=0.0, std=std)

        self.lift = lift
        self.lift_first = lift_first
        self.input_dim = input_dim
        self.m = hidden_dim
        # If this injective MP layer is used as the first layer in the training layer architecture.
        # I.e. we use on injective message passing layer instead of a linear layer
        if self.input_dim > 0:
            self.linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        # Train-free injective message passing, so freeze the parameters of the linear layer and batch norm
        if self.linear:
            for param in self.linear.parameters():
                param.requires_grad = False

    def turn_off_training(self):
        if self.linear:
            for param in self.linear.parameters():
                param.requires_grad = False
            # for param in self.batch_norm.parameters():
            #     param.requires_grad = False

    def turn_on_training(self):
        if self.linear:
            for param in self.linear.parameters():
                param.requires_grad = True
            # for param in self.batch_norm.parameters():
            #     param.requires_grad = True

    def lift_operation(self, h):
        h = self.linear(h)
        h = F.relu(h)
        h = h * torch.tensor(self.m).pow(-0.5)

        return h
    
    """
    A: Adj matrix
    """
    def spectral_norm(self, A):
        return torch.linalg.norm(A, ord=2)


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the custom GNN layer.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, num_node_features].
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges].
            edge_weight (torch.Tensor, optional): Edge weights corresponding to edge_index,
                                                 shape [num_edges]. If None, edges are assumed
                                                 to have weight 1. Defaults to None.

        Returns:
            torch.Tensor: Output node features of shape [num_nodes, num_node_features].
        """            
        num_nodes = x.size(0)
        h = x

        if self.lift_first:
            h = self.lift_operation(h)
            efficients = (1.0 + self.epsilon) * torch.eye(num_nodes) + self.adj_matrix
            h = efficients @ h
            return h



        # 1. Calculate: (1 + epsilon) * I + A
        efficients = (1.0 + self.epsilon) * torch.eye(num_nodes) + self.adj_matrix
        h = efficients @ h

        # do the lift operation if required by input parameters
        if self.lift:
            h = self.lift_operation(h)

        return h

    # def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
    #     """
    #     Constructs messages from source nodes j to target nodes i.
    #     This function is called by `propagate`.

    #     Args:
    #         x_j (torch.Tensor): Features of source nodes j of shape [num_edges, num_node_features].
    #                             These are the features corresponding to the `col` indices in edge_index.
    #         norm (torch.Tensor): The normalization coefficient computed in `forward` for each edge,
    #                              shape [num_edges].

    #     Returns:
    #         torch.Tensor: Messages to be aggregated, shape [num_edges, num_node_features].
    #                       Represents norm_ij * H_j.
    #     """
        # Apply the normalization to the features of the source node (j)
        # norm has shape [num_edges], x_j has shape [num_edges, num_features]
        # We need to reshape norm to [num_edges, 1] for broadcasting
        # return norm.view(-1, 1) * x_j


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
                edge_index,
                num_layers: int,
                num_linear_layers: int,
                input_dim: int,
                mp_hidden_dim: int,
                fl_hidden_dim: int,
                output_dim:int,
                epsilon: float,
                dropout: float = 0.2,
                batch_normalization: bool = False,
                skip_connection: bool = True,
                first_layer_linear: bool = True):
        super().__init__()

        self.num_layers = num_layers
        self.epsilon = epsilon
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.skip_connection = skip_connection
        self.first_layer_linear = first_layer_linear

        self.linear = nn.Linear(input_dim, mp_hidden_dim)
        # lift first
        self.first_injective_layer = InjectiveGNNLayer(edge_index=edge_index, epsilon=self.epsilon, input_dim=input_dim, hidden_dim=mp_hidden_dim)
        # self.batch_norm = BatchNorm(mp_hidden_dim)
        # self.layer_norm = nn.LayerNorm(mp_hidden_dim)
        self.linear2 = nn.Linear(mp_hidden_dim, fl_hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(InjectiveGNNLayer(epsilon=self.epsilon, hidden_dim=mp_hidden_dim, edge_index=edge_index))

        params = list(self.layers.parameters())
        print(f"DecoupleModel __init__ Number of parameters in injective layers: {sum(p.numel() for p in params)}")

        self.linear_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activations = nn.ModuleList()
        for i in range(num_linear_layers):
            self.linear_layers.append(nn.Linear(fl_hidden_dim, fl_hidden_dim))
            self.batch_norms.append(BatchNorm(fl_hidden_dim))
            self.layer_norms.append(nn.LayerNorm(fl_hidden_dim))
            self.dropouts.append(nn.Dropout(self.dropout))
            self.activations.append(nn.ReLU())

        # self.embedding_h = AtomEncoder(emb_dim=fl_hidden_dim)
        # self.embedding_b = BondEncoder(emb_dim=fl_hidden_dim)
        self.classifier = nn.Linear(fl_hidden_dim, output_dim)

    # turn off training for injective message passing layer
    def turn_off_training(self):
        self.first_injective_layer.turn_off_training()
        for i in range(self.num_layers):
            self.layers[i].turn_off_training()


    # turn on training for injective message passing layer
    def turn_on_training(self):
        self.first_injective_layer.turn_on_training()
        for i in range(self.num_layers):
            self.layers[i].turn_on_training()


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the stacked GNN layers.

        Args:
            x (torch.Tensor): Initial node features [num_nodes, num_features].
            edge_index (torch.Tensor): Graph connectivity [2, num_edges].
            edge_weight (torch.Tensor, optional): Edge weights [num_edges]. Defaults to None.

        Returns:
            torch.Tensor: Node features after passing through all layers.
        """
        h = x # Start with initial features

        # Apply the embedding layers
        # h = self.embedding_h(h)

        if self.first_layer_linear:
            h = self.linear(h)
        else:
            h = self.first_injective_layer(h, edge_index)
        # h = self.batch_norm(h)
        # h = self.layer_norm(h)
        # h = torch.relu(h)

        for i, injective_layer in enumerate(self.layers):
            h = injective_layer(h, edge_index, edge_weight=edge_weight)

        # map to feature learning
        h = self.linear2(h)
        for i, linear_layer in enumerate(self.linear_layers):
            pre_h = h
            h = linear_layer(h)
            if self.batch_normalization:
                h = self.batch_norms[i](h)
                # h = self.layer_norms[i](h)
            if self.skip_connection:
                h = h + pre_h
            h = self.activations[i](h)
            if self.dropout > 0:
                h = self.dropouts[i](h)

        # h = global_add_pool(h, batch) if batch is not None else h

        h = self.classifier(h)

        return h
