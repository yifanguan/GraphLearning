import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dataset import load_dataset

import torch
from torch_geometric.utils import get_laplacian, to_undirected


# def embedding_rank(x: torch.Tensor, tol=1e-15) -> int:
#     x_unique = torch.unique(x, dim=0)
#     x_unique = x_unique.double()
#     return torch.linalg.matrix_rank(x_unique, tol=tol)

def embedding_rank(x, tol=1e-15):
    if not torch.isfinite(x).all():
        return float('nan')
    try:
        return torch.linalg.matrix_rank(x.double(), tol=tol).item()
    except RuntimeError:
        return float('nan')

def dirichlet_energy(x, edge_index):
    src, dst = edge_index
    diff = x[src] - x[dst]
    sq_norm = (diff ** 2).sum(dim=1)  # shape: [num_edges]
    # energy = 0.5 * sq_norm.sum() / (x.size(0) * x.size(1))  # normalize by num_nodes * dim

    energy = 0.5 * sq_norm.sum() / ( x.size(1))  # normalize by num_nodes * dim
    return energy.sqrt()

class DeepGNN(nn.Module):
    def __init__(self, gnn_layer_cls, dim, num_layers, act=F.gelu, **layer_kwargs):
        """
        Args:
            gnn_layer_cls: the GNN layer class (e.g., GCNConv, GATConv, etc.),
                           assumed to accept an `act` argument or be wrapped.
            dim: hidden dimension (input/output)
            num_layers: number of GNN layers
            act: nonlinearity (default = GELU)
            **layer_kwargs: passed to each GNN layer
        """
        super().__init__()
        self.layers = nn.ModuleList([
            gnn_layer_cls(dim, dim, act=act, **layer_kwargs)
            for _ in range(num_layers)
        ])

    def forward(self, x, edge_index):
        energies, num_uniques, unique_ranks, norms = [], [], [], []

        for layer in self.layers:
            x = layer(x, edge_index)

            energy = dirichlet_energy(x, edge_index)
            unique_x = torch.unique(x, dim=0)
            rank = embedding_rank(unique_x)
            norm = torch.norm(x) / x.size(0)  # per-node norm

            energies.append(energy)
            num_uniques.append(unique_x.size(0))
            unique_ranks.append(rank)
            norms.append(norm)

        return energies, num_uniques, unique_ranks, norms
    


import torch.nn as nn

class GNNLayerWrapper(nn.Module):
    def __init__(self, gnn_cls, in_dim, out_dim, act=None, **kwargs):
        super().__init__()
        self.gnn = gnn_cls(in_dim, out_dim, **kwargs)
        self.act = act

    def forward(self, x, edge_index):
        x = self.gnn(x, edge_index)
        return self.act(x) if self.act is not None else x


hidden_dim = 50
num_layers = 100  # depth


# Load Cora
data = load_dataset('data', 'Cora')
edge_index = to_undirected(data.edge_index)
# Initialize features as all-ones
x_init = torch.ones((data.num_nodes, hidden_dim))
x_init = data.x
hidden_dim = data.x.shape[1]
# Instantiate and run

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch.nn import Sequential, Linear, GELU

# model = DeepGNN(iMP, dim=hidden_dim, num_layers=100, act=F.relu)

### wrappers
# wrapped_gnn_cls = lambda in_dim, out_dim, act=None, **kwargs: GNNLayerWrapper(GCNConv, in_dim, out_dim, act=act, **kwargs)

wrapped_gnn_cls = lambda in_dim, out_dim, act=None, **kwargs: GNNLayerWrapper(GATConv, in_dim, out_dim, act=act, **kwargs)

###
model = DeepGNN(wrapped_gnn_cls, dim=hidden_dim, num_layers=100, act=F.relu)

###
# model = DeepGNN(GATConv, dim=hidden_dim, num_layers=100, heads=1, concat=False)

# model = DeepGNN(SAGEConv, dim=hidden_dim, num_layers=100)

# def build_gin_mlp(hidden_dim):
#     return Sequential(Linear(hidden_dim, hidden_dim), GELU(), Linear(hidden_dim, hidden_dim))

# GINLayer = lambda in_dim, out_dim: GINConv(nn=build_gin_mlp(in_dim), train_eps=True)
# model = DeepGNN(GINLayer, dim=hidden_dim, num_layers=100)

with torch.no_grad():
    energies, num_uniques, unique_ranks, norms = model(x_init, edge_index)

# Print results every 10 layers
for l, (e, u, r, n) in enumerate(zip(energies, num_uniques, unique_ranks, norms), 1):
    if l % 10 == 0 or l == 1:
        print(f"Layer {l:03d}: Dirichlet Energy = {e}, Unique Embeddings = {u}, Unique Rank = {r}, per-node norm = {n}")