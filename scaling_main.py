from utils.over_smoothing_measure import DeepGNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch.nn import Sequential, Linear, GELU
from utils.dataset import load_dataset
from torch_geometric.utils import to_undirected

data = load_dataset('Cora')
edge_index = to_undirected(data.edge_index)

hidden_dim = 50
num_layers = 100  # depth

# Initialize features as all-ones
x_init = torch.ones((data.num_nodes, hidden_dim))

# Instantiate and run


# model = DeepGNN(GCNConv, dim=hidden_dim, num_layers=100)

# model = DeepGNN(GATConv, dim=hidden_dim, num_layers=100, heads=1, concat=False)

# model = DeepGNN(SAGEConv, dim=hidden_dim, num_layers=100)

def build_gin_mlp(hidden_dim):
    return Sequential(Linear(hidden_dim, hidden_dim), GELU(), Linear(hidden_dim, hidden_dim))

GINLayer = lambda in_dim, out_dim: GINConv(nn=build_gin_mlp(in_dim), train_eps=True)
model = DeepGNN(GINLayer, dim=hidden_dim, num_layers=100)

with torch.no_grad():
    energies, num_uniques, unique_ranks = model(x_init, edge_index)

# Print results every 10 layers
for l, (e, u, r) in enumerate(zip(energies, num_uniques, unique_ranks), 1):
    if l % 10 == 0:
        print(f"Layer {l:03d}: Dirichlet Energy = {e:.4f}, Unique Embeddings = {u}, Unique Rank = {r}")



plt.style.use('seaborn-v0_8-muted')  # Clean aesthetic

plt.figure(figsize=(10, 4))
plt.plot(range(1, len(energies) + 1), energies,
         marker='o', markersize=4, markerfacecolor='white',
         linewidth=1.5, label='Dirichlet Energy')
plt.title("Dirichlet Energy vs. GIN Layer Depth", fontsize=14)
plt.xlabel("Layer", fontsize=12)
plt.ylabel("Dirichlet Energy", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(range(1, len(num_uniques) + 1), num_uniques,
         marker='D', markersize=4, markerfacecolor='white',
         linewidth=1.5, label='Unique Embeddings')
plt.plot(range(1, len(unique_ranks) + 1), unique_ranks,
         marker='s', markersize=4, markerfacecolor='white',
         linewidth=1.5, label='Unique Rank')
plt.title("# Unique Embeddings and Rank vs. GIN Layer Depth", fontsize=14)
plt.xlabel("Layer", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()