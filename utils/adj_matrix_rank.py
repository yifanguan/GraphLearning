import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import sys, os
# Add parent folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

dataset_name = 'Cora'

current_file = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file))
dataset = Planetoid(root=f'{parent_dir}/data/Planetoid', name=dataset_name, transform=T.NormalizeFeatures())
data = dataset[0]  # Cora has only one graph

###############
tolerance=1e-5
num_mp_layer=6
###############

def calculate_adj_matrix_rank(data):
    adj_dense = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]  # shape: [N, N]
    rank = torch.linalg.matrix_rank(adj_dense, tol=tolerance)

    return rank.item()

for tol in [1e-5, 1e-6, 1e-7]:
    tolerance = tol
    rank = calculate_adj_matrix_rank(data)
    print(f"When tolerance is {tolerance}, Rank of adjacency matrix: {rank}")
