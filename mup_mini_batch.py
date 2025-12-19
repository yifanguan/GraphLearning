import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv
# https://pytorch-geometric.readthedocs.io/en/2.5.2/tutorial/neighbor_loader.html
# https://medium.com/stanford-cs224w/a-tour-of-pygs-data-loaders-9f2384e48f8f
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import Compose, NormalizeFeatures, RandomNodeSplit
import random
import numpy as np
from mup_impl.mup import init_mup_input, init_mup_hidden, init_mup_readout, mup_param_groups, MuGNN, make_mup_optimizer
from mup_impl.train import train_val_test_mask_helper, balance_dataset, train_loop
from utils.dataset import load_dataset, load_large_dataset
from torch_geometric.utils import to_undirected, add_self_loops
from torch_geometric.loader import RandomNodeLoader, NeighborLoader
import torch_geometric.transforms as T
import pandas as pd
from utils.timestamp import get_timestamp
from functools import partial
import seaborn as sns
from matplotlib.lines import Line2D
from mup_impl.plot_utils import plot_best_metric, plot_loss, save_results
import mup_impl.plot_utils as pu
import time

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# === experiment args === (TODO: make them a arg list when needed including hyperparameters)
folder_name = 'mup_ogbn_arxiv_mini_batch_sgd_new_version'
dataset_name = 'ogbn-arxiv'
# dataset_name = 'ogbn-products'
# dataset_name = 'cora'
# dataset_name = 'citeseer'
# dataset_name = 'wikics'
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
# data = data.to(device) # Nov 9th, not necessary because batch will be moved to gpu

# balance dataset, only select data class with enough data
balance_dataset(dataset, K=10)

data = dataset.graph

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
widths = [256]
# depths = [1,2,4,6,8,10]
# [0,1,2,4,8,16]
depths = [1,2,4,8,16]
# depths = [4]
lrs    = np.linspace(-11, 1, 9)   # add/remove as you like
# lrs = np.linspace(-8, 1, 10) # add/remove as you like

sgc_k = 2
num_epochs = 10
log_every = 2

# === Placeholder for results ===
results = {}

# === Define loss function ===
criterion = torch.nn.CrossEntropyLoss()

# === Mini-batch Data loaders ===
train_loader = None
val_loader = None
test_loader = None
if True:
    # Set split indices to masks.
    if dataset.train_mask.dtype != torch.bool: 
        for split in ['train', 'val', 'test']:
            mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
            mask[getattr(dataset, f'{split}_idx')] = True
            data[f'{split}_mask'] = mask
    else:
        for split in ['train', 'val', 'test']: 
            data[f'{split}_mask'] = getattr(dataset, f'{split}_idx')

    # Neighbor sampling parameters
    num_neighbors = [10] * sgc_k # sample 10 neighbors per layer (2-hop)
    batch_size = 2048

    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=data.train_mask,
        batch_size=batch_size,
        shuffle=True,
        num_workers=5,
        persistent_workers=True,
    )

    test_batch_size = 2048
    # Validation loader: full neighbors (no sampling) or same as train for speed.
    val_loader = NeighborLoader(
        data,
        # num_neighbors=[-1] * sgc_k,        # use full neighbors for val
        num_neighbors=[-1],
        input_nodes=data.val_mask,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=5,
        persistent_workers=True,
    )

    test_loader = NeighborLoader(
        data,
        # num_neighbors=[-1] * sgc_k,   # full neighbors for eval (no sampling)
        num_neighbors=[-1],
        input_nodes=data.test_mask,     # all nodes
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=5,
        persistent_workers=True,
    )
# else:
#     # Neighbor sampling parameters
#     num_neighbors = [10] * k # sample 10 neighbors per layer (2-hop)
#     batch_size = 1024
#     data

#     train_loader = NeighborLoader(
#         data,
#         num_neighbors=num_neighbors,
#         input_nodes=data.train_mask,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=5,
#         persistent_workers=True,
#     )

#     val_loader = NeighborLoader(
#         data,
#         num_neighbors=num_neighbors,
#         input_nodes=data.train_mask,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=5,
#         persistent_workers=True,
#     )

#     test_loader = NeighborLoader(
#         data,
#         num_neighbors=[-1],   # full neighbors for eval (no sampling)
#         input_nodes=None,     # all nodes
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=5,
#         persistent_workers=True,
#     )


transform = T.Compose([T.ToDevice(device), T.ToSparseTensor()])
# transform = T.Compose([T.ToDevice(device)])


# === Training function ===

#### full batch train #####
# def train(model, data, dataset, optimizer):
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index)
#     # loss = criterion(out[data.train_mask], data.y[data.train_mask])
#     loss = criterion(out[dataset.train_idx], data.y[dataset.train_idx])
#     loss.backward()
#     optimizer.step()
#     return loss.item()
#### END full batch train #####


# === Mini-batch training ===
# def train(model, train_loader, optimizer, device):
#     model.train()
#     total_loss = 0
#     total_examples = 0

#     for batch in train_loader:
#         batch = batch.to(device)
#         optimizer.zero_grad()

#         out = model(batch.x, batch.edge_index)
#         # only first batch_size nodes are seeds in NeighborLoader
#         # seed_nodes = batch.n_id[:batch.batch_size]
#         num_seeds = batch.input_id.numel()
#         loss = criterion(out[:num_seeds], batch.y[:num_seeds].view(-1))

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * num_seeds
#         total_examples += num_seeds

#     return total_loss / total_examples
# === END Mini-batch training ===


##########Profiling version train USE when needed###########
def train(model, train_loader, optimizer, device):
    start = time.perf_counter()
    model.train()
    total_loss = 0
    total_examples = 0

    sampling_times = []
    train_times = []

    # -------- start profiling --------
    loader_iter = iter(train_loader)

    while True:
        # ============================
        #   1. measure SAMPLING time
        # ============================
        t0 = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        sampling_time = time.perf_counter() - t0
        sampling_times.append(sampling_time)

        # ============================
        #   2. measure GPU TRAIN time
        # ============================
        batch = batch.to(device)

        torch.cuda.synchronize()       # sync before GPU timing
        t1 = time.perf_counter()

        out = model(batch.x, batch.edge_index)
        num_seeds = batch.input_id.numel()
        loss = criterion(out[:num_seeds], batch.y[:num_seeds].view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()       # sync after GPU timing
        train_time = time.perf_counter() - t1
        train_times.append(train_time)

        # compute total loss
        total_loss += loss.item() * num_seeds
        total_examples += num_seeds

    # Print profiling stats
    print(
        f"[PROFILE] sampling avg: {np.mean(sampling_times):.4f} s | "
        f"train avg: {np.mean(train_times):.4f} s | "
        f"ratio sampling/train = {np.mean(sampling_times) / np.mean(train_times):.2f}x"
    )
    end = time.perf_counter()
    print(
        f"train total time: {(end-start):.4f} s | "
    )
    return total_loss / total_examples
##########END Profiling version train ###########

#### plain mini-batch version #####
# === Evaluation function ===
# @torch.no_grad()
# def evaluate(model, data, dataset):
#     model.eval()
#     out = model(data.x, data.edge_index)
#     result = {}
#     for split in ['train', 'val', 'test']:
#         mask = getattr(dataset, f"{split}_mask")
#         loss = criterion(out[mask], data.y[mask]).item()
#         pred = out[mask].argmax(dim=1)
#         acc = (pred == data.y[mask]).sum().item() / len(pred)
#         result[f"{split}_loss"] = loss
#         result[f"{split}_acc"] = acc
#     return result
#### END plain mini-batch version #####


##### profile mini-batch version######
@torch.no_grad()
def evaluate(model, loaders, device):
    model.eval()
    results = {}

    start_total = time.perf_counter()

    for split, loader in loaders.items():
        total_loss = 0
        total_examples = 0
        y_true, y_pred = [], []

        # --- profiling variables ---
        sampling_times = []
        compute_times = []

        # turn loader into manual iterator
        loader_iter = iter(loader)

        while True:
            # ------------------------------------
            # 1. Profile SAMPLING (next(loader))
            # ------------------------------------
            t0 = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            t1 = time.perf_counter()
            sampling_times.append(t1 - t0)

            # ------------------------------------
            # 2. Profile COMPUTE
            # ------------------------------------
            batch = batch.to(device)
            t2 = time.perf_counter()
            out = model(batch.x, batch.edge_index)
            t3 = time.perf_counter()
            compute_times.append(t3 - t2)

            # ------------------------------------
            # 3. Evaluation logic
            # ------------------------------------
            num_seeds = batch.input_id.numel()
            logits = out[:num_seeds]
            preds = logits.argmax(dim=1)
            labels = batch.y[:num_seeds].view(-1)

            loss = criterion(logits, labels)
            total_loss += loss.item() * num_seeds
            total_examples += num_seeds
            y_true.append(labels.cpu())
            y_pred.append(preds.cpu())

        # concatenate predictions
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        acc = (y_true == y_pred).float().mean().item()

        # store results
        results[f"{split}_loss"] = total_loss / total_examples
        results[f"{split}_acc"] = acc

        # print profile for this split
        if len(sampling_times) > 0:
            print(
                f"[PROFILE] split={split} | "
                f"sampling avg: {sum(sampling_times)/len(sampling_times):.4f}s | "
                f"compute avg:  {sum(compute_times)/len(compute_times):.4f}s | "
                f"ratio sampling/compute = "
                f"{(sum(sampling_times)/len(sampling_times)) / (sum(compute_times)/len(compute_times)):.2f}x"
            )

    end_total = time.perf_counter()
    print(f"evaluate total time: {(end_total - start_total):.4f}s")

    return results
##### END profile mini-batch version######


#### full batch evaluation #####
# @torch.no_grad()
# def evaluate(model, data, dataset):
#     start = time.perf_counter()
#     model.eval()
#     out = model(data.x.to(device), data.edge_index.to(device))
#     result = {}
#     for split in ['train', 'val', 'test']:
#         mask = getattr(dataset, f"{split}_mask")
#         loss = criterion(out[mask], data.y[mask]).item()
#         pred = out[mask].argmax(dim=1)
#         acc = (pred == data.y[mask]).sum().item() / len(pred)
#         result[f"{split}_loss"] = loss
#         result[f"{split}_acc"] = acc

#     end = time.perf_counter()
#     print(
#         f"evaluate total time: {(end-start):.4f} s | "
#     )

#     return result
#### END full batch evaluation #####


# === Main experiment loop ===
train_func = partial(train, train_loader=train_loader, device=device)
evaluate_func = partial(evaluate, loaders={'train' : train_loader, 'val' : val_loader, 'test' : test_loader}, device=device)
rows = train_loop(widths, depths, lrs, input_dim=d, output_dim=dataset.num_classes, K=sgc_k, device=device,
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
