import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import get_laplacian, to_undirected, add_self_loops, degree

# def dirichlet_energy(X, edge_index):
#     '''
#     Compute Dirichlet energy: 1/v * sum_{i in v} sum_{j in N_i} ||x_i - x_j||^2
#     X: node hidden features, num nodes * hidden dim
#     '''
#     row, col = edge_index
#     diff = X[row] - X[col]
#     energy = (diff ** 2).sum()

#     return energy / X.shape[0]


# def dirichlet_energy(x, edge_index):
#     src, dst = edge_index
#     diff = x[src] - x[dst]
#     sq_norm = (diff ** 2).sum(dim=1)
#     out = 0.5 * sq_norm.sum() / x.size(0) # divide by hidden_dim / x.size(1
#     return out**0.5

def dirichlet_energy(x, edge_index):
    '''
    normalized version: normalized by number of nodes
    '''
    src, dst = edge_index
    diff = x[src] - x[dst]
    sq_norm = (diff ** 2).sum(dim=1)
    loss = sq_norm.sum() / x.size(0)
    return loss





def dirichlet_energy(x, edge_index):
    """
    Computes the symmetric normalized Dirichlet energy:
        sum_{(i,j)} || x_i/sqrt(d_i) - x_j/sqrt(d_j) ||^2
    with self-loops added to avoid zero degree nodes.

    Args:
        x (Tensor): Node embeddings of shape [num_nodes, feature_dim].
        edge_index (LongTensor): Edge index of shape [2, num_edges].

    Returns:
        float: Normalized Dirichlet energy.
    """
    if not torch.isfinite(x).all():
        return float('nan')

    num_nodes = x.size(0)

    # Add self-loops to ensure non-zero degrees
    # edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes, dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg == 0] = 0.0  # just in case

    # No need to check deg == 0 anymore due to self-loops
    x_norm = x * deg_inv_sqrt.unsqueeze(-1)

    diff = x_norm[row] - x_norm[col]         # [num_edges, feature_dim]
    sq_dist = (diff ** 2).sum(dim=-1)        # [num_edges]
    energy = sq_dist.sum()
    if not torch.isfinite(energy):
        return float('nan')
    return energy

def normalized_dirichlet_energy(x, edge_index, energy=None):
    energy = dirichlet_energy(x, edge_index) if energy is None else energy
    denom = x.pow(2).sum()
    return (energy / denom).item() if denom > 0 else float('nan')



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

