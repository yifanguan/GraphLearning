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
from mup_impl.train import train_val_test_mask_helper
import pandas as pd
from matplotlib.lines import Line2D
import seaborn as sns
from utils.timestamp import get_timestamp
from mup_impl.plot_utils import plot_best_metric, plot_loss, save_results
import mup_impl.plot_utils as pu

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# === experiment args === (TODO: make them a arg list when needed including hyperparameters)
dataset_name = 'ogbn-arxiv'
folder_name = 'mup_arxiv_sgd_full_batch2'
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

num_epochs = 4000
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
rows = []
for width in widths:
    for depth in depths:
        for log2lr in lrs:
            lr = 2**log2lr
            key = f"width={width} depth={depth} lr={lr:g}"
            print(f"\n=== Training {key} ===")

            model = MuGNN(
                input_dim=d, # dataset.num_node_features
                hidden_dim=width,
                output_dim=dataset.num_classes, # dataset.num_classes
                num_fc_layers=depth, # depth = num_fc_layers + 1; actually, so we minus one here
                K=2).to(device)

            optimizer = make_mup_optimizer(model, base_lr=lr, opt="sgd", weight_decay=0.0)

            # Best trackers (value + epoch)
            best = {
                "train_loss": (math.inf, -1),
                "val_loss":   (math.inf, -1),
                "test_loss":  (math.inf, -1),
                "train_acc":  (0.0, -1),
                "val_acc":    (0.0, -1),
                "test_acc":   (0.0, -1),
            }

            # loss trackers
            # we evaluate val and test every log_every epoch, so record these loss point for val and test loss plot
            # we evaluate train every epoch (for best train loss, we get its value from evaluate step, the train loss plot is
            # diverged from this observation for more data points purpose) change this part if needed.
            loss_dict = {
                "train_loss": [],
                "val_loss": [],
                "test_loss": []
            }

            for epoch in range(1, num_epochs + 1):
                train_loss = train(model, data, dataset, optimizer)
                loss_dict['train_loss'].append((train_loss, epoch))

                if epoch == 1 or epoch % log_every == 0 or epoch == num_epochs:
                    m = evaluate(model, data, dataset) # result dict

                    # update bests
                    for k in ["train_loss", "val_loss", "test_loss"]:
                        if m[k] < best[k][0]:
                            best[k] = (m[k], epoch)
                    for k in ["train_acc", "val_acc", "test_acc"]:
                        if m[k] > best[k][0]:
                            best[k] = (m[k], epoch)

                    # record loss
                    loss_dict['val_loss'].append((m['val_loss'], epoch))
                    loss_dict['test_loss'].append((m['test_loss'], epoch))

                    print(
                        f"Epoch {epoch:03d} | "
                        f"Train: loss {m['train_loss']:.4f}, acc {m['train_acc']:.4f}, best {best['train_acc'][0]:.4f} (ep {best['train_acc'][1]}) | "
                        f"Val: loss {m['val_loss']:.4f}, acc {m['val_acc']:.4f}, best {best['val_acc'][0]:.4f} (ep {best['val_acc'][1]}) | "
                        f"Test: loss {m['test_loss']:.4f}, acc {m['test_acc']:.4f}, best {best['test_acc'][0]:.4f} (ep {best['test_acc'][1]})"
                    )

            # last epoch metrics
            last_m = evaluate(model, data, dataset)

            # store a row per run
            rows.append({
                "width": width,
                "depth": depth,
                "lr": lr,
                "best_train_loss": best["train_loss"][0],
                "best_train_loss_epoch": best["train_loss"][1],
                "best_val_loss": best["val_loss"][0],
                "best_val_loss_epoch": best["val_loss"][1],
                "best_test_loss": best["test_loss"][0],
                "best_test_loss_epoch": best["test_loss"][1],
                "best_train_acc": best["train_acc"][0],
                "best_train_acc_epoch": best["train_acc"][1],
                "best_val_acc": best["val_acc"][0],
                "best_val_acc_epoch": best["val_acc"][1],
                "best_test_acc": best["test_acc"][0],
                "best_test_acc_epoch": best["test_acc"][1],
                "train_loss": loss_dict['train_loss'],
                "val_loss": loss_dict['val_loss'],
                "test_loss": loss_dict['test_loss'],
                "last_train_loss": last_m["train_loss"],
                "last_val_loss": last_m["val_loss"],
                "last_test_loss": last_m["test_loss"],
                "last_train_acc": last_m["train_acc"],
                "last_val_acc": last_m["val_acc"],
                "last_test_acc": last_m["test_acc"],
            })

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
