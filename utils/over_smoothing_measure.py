import torch
import torch.nn as nn
import torch.nn.functional as F

# def dirichlet_energy(X, edge_index):
#     '''
#     Compute Dirichlet energy: 1/v * sum_{i in v} sum_{j in N_i} ||x_i - x_j||^2
#     X: node hidden features, num nodes * hidden dim
#     '''
#     row, col = edge_index
#     diff = X[row] - X[col]
#     energy = (diff ** 2).sum()

#     return energy / X.shape[0]


def dirichlet_energy(x, edge_index):
    src, dst = edge_index
    diff = x[src] - x[dst]
    sq_norm = (diff ** 2).sum(dim=1)
    out = 0.5 * sq_norm.sum() / x.size(1) / x.size(0) # divide by hidden_dim
    return out**0.5



# def dirichlet_energy(X, edge_index, edge_weight=None):
#     """
#     Compute Dirichlet energy: sum_{i,j in edges} w_ij * ||x_i - x_j||^2
#     - X: (N, d) node features
#     - edge_index: (2, E) tensor of edges
#     - edge_weight: (E,) tensor of edge weights (default 1 if None)
#     """
#     row, col = edge_index
#     diff = X[row] - X[col]                     # (E, d)
#     sq_dist = (diff ** 2).sum(dim=1)           # (E,)
#     if edge_weight is not None:
#         energy = (edge_weight * sq_dist).sum()
#     else:
#         energy = sq_dist.sum()
#     return energy


def embedding_rank(x, tol=1e-15):
    if not torch.isfinite(x).all():
        return float('nan')
    try:
        return torch.linalg.matrix_rank(x.double(), tol=tol).item()
    except RuntimeError:
        return float('nan')

class DeepGNN(nn.Module):
    def __init__(self, gnn_layer_cls, dim, num_layers, activation=F.gelu, **layer_kwargs):
        """
        Args:
            gnn_layer_cls: the GNN layer class (e.g., GCNConv, GATConv, etc.)
            dim: hidden dimension (input/output)
            num_layers: number of GNN layers
            activation: nonlinearity (default = GELU)
            **layer_kwargs: passed to each GNN layer
        """
        super().__init__()
        self.layers = nn.ModuleList([
            gnn_layer_cls(dim, dim, **layer_kwargs)
            for _ in range(num_layers)
        ])
        self.activation = activation

    def forward(self, x, edge_index):
        energies, num_uniques, unique_ranks = [], [], []

        for layer in self.layers:
            x = layer(x, edge_index)
            if self.activation is not None:
                x = self.activation(x)

            energy = dirichlet_energy(x, edge_index)
            unique_x = torch.unique(x, dim=0)
            rank = embedding_rank(unique_x)

            energies.append(energy)
            num_uniques.append(unique_x.size(0))
            unique_ranks.append(rank)

        return energies, num_uniques, unique_ranks

