import torch


def dirichlet_energy(X, edge_index):
    '''
    Compute Dirichlet energy: 1/v * sum_{i in v} sum_{j in N_i} ||x_i - x_j||^2
    X: node hidden features, num nodes * hidden dim
    '''
    row, col = edge_index
    diff = X[row] - X[col]
    energy = (diff ** 2).sum()

    return energy / X.shape[0]

    



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