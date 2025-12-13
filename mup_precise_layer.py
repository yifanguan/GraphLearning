import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import Compose, NormalizeFeatures, RandomNodeSplit
import random
import numpy as np
from utils.dataset import load_dataset, load_large_dataset
from torch_geometric.utils import to_undirected, add_self_loops
from mup_impl.mup import init_mup_hidden, init_mup_input, init_mup_readout, mup_param_groups, MuGNN, make_mup_optimizer
from mup_impl.train import train_val_test_mask_helper, train_loop
import pandas as pd
from matplotlib.lines import Line2D
import seaborn as sns
from utils.timestamp import get_timestamp
from mup_impl.plot_utils import plot_best_metric, plot_loss, save_results
import mup_impl.plot_utils as pu
from functools import partial

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# === experiment args === (TODO: make them a arg list when needed including hyperparameters)
dataset_name = 'ogbn-arxiv'
folder_name = 'mup_arxiv_sgd_full_batch3'
# dataset_name = 'ogbn-products'
# dataset_name = 'cora'
# dataset_name = 'citeseer'
# dataset_name = 'wikics'


# === Dataset ===
dataset = load_dataset(data_dir='data', dataset_name=dataset_name)

display_step = 10

d = dataset.graph.x.shape[1]
c = dataset.label.max().item() + 1
n = dataset.graph.x.shape[0]
assert c == dataset.num_classes
print(f'Dataset: {dataset_name}, num nodes: {n}, num node features: {d}, num classes: {c}')
dataset.graph.edge_index = to_undirected(dataset.graph.edge_index)
dataset.graph.edge_index, _ = add_self_loops(dataset.graph.edge_index, num_nodes=n)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset.graph
data = data.to(device)

train_val_test_mask_helper(dataset_name, dataset)

print('mask sum')
print(f"# Train: {dataset.train_mask.sum().item()}")
print(f"# Val: {dataset.val_mask.sum().item()}")
print(f"# Test: {dataset.test_mask.sum().item()}")

print('mask length')
print(f"# Train: {len(dataset.train_mask)}")
print(f"# Val: {len(dataset.val_mask)}")
print(f"# Test: {len(dataset.test_mask)}")


# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Hyperparameters ===
# widths = [128, 256, 512, 1024]
widths = [512]
# depths = [1,2,4,6,8,10]
# [0,1,2,4,8,16]
depths = [1,2,4,8,16]
# depths = [4]
lrs    = np.linspace(-7, 3, 11)   # add/remove as you like

num_epochs = 4
log_every = 10

# === Placeholder for results ===
results = {}

# === Define loss function ===
criterion = torch.nn.CrossEntropyLoss()

# === Training function ===
def train(model, data, dataset, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    # loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss = criterion(out[dataset.train_idx], data.y[dataset.train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

# === Evaluation function ===
@torch.no_grad()
def evaluate(model, data, dataset):
    model.eval()
    out = model(data.x, data.edge_index)
    result = {}
    for split in ['train', 'val', 'test']:
        mask = getattr(dataset, f"{split}_mask")
        loss = criterion(out[mask], data.y[mask]).item()
        pred = out[mask].argmax(dim=1)
        acc = (pred == data.y[mask]).sum().item() / len(pred)
        result[f"{split}_loss"] = loss
        result[f"{split}_acc"] = acc
    return result

# === Main experiment loop ===
train_func = partial(train, data=data, dataset=dataset)
evaluate_func = partial(evaluate, data=data, dataset=dataset)
rows = train_loop(widths, depths, lrs, input_dim=d, output_dim=dataset.num_classes, K=2, device=device,
                  num_epochs=num_epochs, log_every=log_every, train_func=train_func, evaluate_func=evaluate_func)

# Save Results
save_results(rows, folder_name, dataset_name)

pu.folder_name = folder_name
pu.dataset_name = dataset_name

# --------------------------
# Results table
# --------------------------
df = pd.DataFrame(rows).sort_values(["depth", "width", "lr"]).reset_index(drop=True)
# print("\n=== BEST METRICS (per run) ===")
# print(df.to_string(index=False))

# --------------------------
# Optional: compact plots for best metrics
# --------------------------
sns.set(style="whitegrid")

plot_best_metric(df, "best_train_loss", "Best Train Loss vs LR", log_y=True)
plot_best_metric(df, "best_val_loss",  "Best Val Loss vs LR",   log_y=True)
plot_best_metric(df, "best_test_loss", "Best Test Loss vs LR", log_y=True)

plot_best_metric(df, "best_train_acc", "Best Train Accuracy vs LR", log_y=False)
plot_best_metric(df, "best_val_acc",  "Best Val Accuracy vs LR",   log_y=False)
plot_best_metric(df, "best_test_acc", "Best Test Accuracy vs LR", log_y=False)

plot_best_metric(df, "train_loss", "Last Train Loss vs LR", log_y=True, use_last=True)
plot_best_metric(df, "val_loss",   "Last Val Loss vs LR",   log_y=True, use_last=True)
plot_best_metric(df, "test_loss",  "Last Test Loss vs LR",  log_y=True, use_last=True)

plot_best_metric(df, "train_acc", "Last Train Accuracy vs LR", log_y=False, use_last=True)
plot_best_metric(df, "val_acc",   "Last Val Accuracy vs LR",   log_y=False, use_last=True)
plot_best_metric(df, "test_acc",  "Last Test Accuracy vs LR",  log_y=False, use_last=True)


# plot loss
lrs = sorted(pd.to_numeric(df["lr"], errors="coerce").dropna().unique())
for lr in lrs:
    plot_loss(df, lr, f'Loss vs epoch for LR: {lr}', log_y=True)
