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
import math
from mup_impl.mup import init_mup_input, init_mup_hidden, init_mup_readout, mup_param_groups, MuGNN

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def make_mup_optimizer(model, base_lr, opt="adam", weight_decay=0.0, momentum=0.9, betas=(0.9, 0.999)):
    groups = mup_param_groups(model, base_lr, opt, weight_decay)
    if opt == "sgd":
        return torch.optim.SGD(groups, lr=base_lr, momentum=momentum)
    else:
        return torch.optim.Adam(groups, lr=base_lr, betas=betas)


# experiment code:

# from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import Compose, NormalizeFeatures, RandomNodeSplit

# transform = Compose([
#     NormalizeFeatures(),
#     RandomNodeSplit(num_train_per_class=0.6, num_val=0.2, num_test=0.2, split='train_rest')
#     # RandomNodeSplit(split="random", num_train_per_class=20, num_val=500, num_test=1000)
# ])

# # dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=transform)
# dataset = Planetoid(root='/tmp/PubMed', name='PubMed', transform=transform)
# # dataset = Amazon(root='/tmp/Amazon', name='Computers', transform=transform)
# # dataset = Amazon(root='/tmp/Amazon', name='Photo', transform=transform)
# # dataset = Coauthor(root='/tmp/Coauthor', name='Physics', transform=transform)
# # dataset = Coauthor(root='/tmp/Coauthor', name='CS', transform=transform)
# # dataset = WikiCS(root='/tmp/WikiCS', transform=transform)
# data = dataset[0]

# # Device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
# data = data.to(device)
# print(f"# Train: {data.train_mask.sum().item()}")
# print(f"# Val: {data.val_mask.sum().item()}")
# print(f"# Test: {data.test_mask.sum().item()}")



def train_val_test_mask_helper(dataset_name, dataset):
    '''
    UPDATE dataset masks and idx for experiment.
    This is the single place for changing them for simplicity.
    Customized for each dataset.
    '''
    if dataset_name == 'pubmed' or dataset_name == 'cora' or dataset_name == 'citeseer' or dataset_name == 'ogbn-products':
        transform = RandomNodeSplit(num_train_per_class=0.6, num_val=0.2, num_test=0.2, split='train_rest')
        dataset.graph = transform(dataset.graph)
        dataset.train_idx = dataset.graph.train_mask
        dataset.train_mask = dataset.graph.train_mask
        dataset.val_idx = dataset.graph.val_mask
        dataset.val_mask = dataset.graph.val_mask
        dataset.test_idx = dataset.graph.test_mask
        dataset.test_mask = dataset.graph.test_mask

    return dataset



from utils.dataset import load_dataset, load_large_dataset
from torch_geometric.utils import to_undirected, add_self_loops
# dataset_name = 'ogbn-arxiv'
dataset_name = 'ogbn-products'
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
data = dataset.graph
# data = data.to(device) # Nov 9th, not necessary because batch will be moved to gpu

train_val_test_mask_helper(dataset_name, dataset)

print('mask sum')
print(f"# Train: {dataset.train_mask.sum().item()}")
print(f"# Val: {dataset.val_mask.sum().item()}")
print(f"# Test: {dataset.test_mask.sum().item()}")

print('mask length')
print(f"# Train: {len(dataset.train_mask)}")
print(f"# Val: {len(dataset.val_mask)}")
print(f"# Test: {len(dataset.test_mask)}")

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import Compose, NormalizeFeatures, RandomNodeSplit
import random
import numpy as np
import math

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Dataset ===
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# data = dataset[0].to(device)

# === Hyperparameters ===
# widths = [128, 256, 512, 1024]
widths = [256]
# depths = [1,2,4,6,8,10]
# [0,1,2,4,8,16]
depths = [0,1,2,4,8,16]
# depths = [4]
# lrs    = np.linspace(-11, 1, 15)   # add/remove as you like
lrs = np.linspace(-10, -3, 10) # add/remove as you like

num_epochs = 10
log_every = 10

# === Placeholder for results ===
results = {}

# === Define loss function ===
criterion = torch.nn.CrossEntropyLoss()

sgc_k = 2
from torch_geometric.loader import RandomNodeLoader, NeighborLoader
import torch_geometric.transforms as T
train_loader = None
val_loader = None
test_loader = None
# if dataset_name == 'ogbn-products':
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
# def train(model, data, dataset, optimizer):
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index)
#     # loss = criterion(out[data.train_mask], data.y[data.train_mask])
#     loss = criterion(out[dataset.train_idx], data.y[dataset.train_idx])
#     loss.backward()
#     optimizer.step()
#     return loss.item()

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
#         loss = criterion(out[:num_seeds], batch.y[:num_seeds])

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * num_seeds
#         total_examples += num_seeds

#     return total_loss / total_examples

##########Profiling version USE when needed###########
import time

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
        loss = criterion(out[:num_seeds], batch.y[:num_seeds])

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
##########Profiling version###########

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
            labels = batch.y[:num_seeds]

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
#### full batch evaluation #####


# === Main experiment loop ===
rows = []
folder_name = 'mup_ogbn_products_mini_batch'

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
                K=sgc_k).to(device)

            optimizer = make_mup_optimizer(model, base_lr=lr, opt="adam", weight_decay=0.0)

            # Best trackers (value + epoch)
            best = {
                "train_loss": (math.inf, -1),
                "val_loss":   (math.inf, -1),
                "test_loss":  (math.inf, -1),
                "train_acc":  (0.0, -1),
                "val_acc":    (0.0, -1),
                "test_acc":   (0.0, -1),
            }

            for epoch in range(1, num_epochs + 1):
                # train_loss = train(model, data, dataset, optimizer)
                train_loss = train(model, train_loader, optimizer, device)

                if epoch == 1 or epoch % log_every == 0 or epoch == num_epochs:
                    # m = evaluate(model, data, dataset) # result dict
                    m = evaluate(model, {'train' : train_loader, 'val' : val_loader, 'test' : test_loader}, device) # result dict

                    # update bests
                    for k in ["train_loss", "val_loss", "test_loss"]:
                        if m[k] < best[k][0]:
                            best[k] = (m[k], epoch)
                    for k in ["train_acc", "val_acc", "test_acc"]:
                        if m[k] > best[k][0]:
                            best[k] = (m[k], epoch)

                    print(
                        f"Epoch {epoch:03d} | "
                        f"Train: loss {m['train_loss']:.4f}, acc {m['train_acc']:.4f}, best {best['train_acc'][0]:.4f} (ep {best['train_acc'][1]}) | "
                        f"Val: loss {m['val_loss']:.4f}, acc {m['val_acc']:.4f}, best {best['val_acc'][0]:.4f} (ep {best['val_acc'][1]}) | "
                        f"Test: loss {m['test_loss']:.4f}, acc {m['test_acc']:.4f}, best {best['test_acc'][0]:.4f} (ep {best['test_acc'][1]})"
                    )


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
            })


import pandas as pd
from utils.timestamp import get_timestamp

# record full experiment results
df = pd.DataFrame(rows).sort_values(["depth", "width", "lr"]).reset_index(drop=True)
csv_path = f"{folder_name}/{dataset_name}_all_runs_{get_timestamp()}.csv"
df.to_csv(csv_path, index=False)

print(f"\nSaved all experiment runs to: {csv_path}")
print(df.head())

# aggregate results among (width, depth) pairs, and record the results
# ============================================================
# Aggregate best accuracies across LRs
# ============================================================
summary_df = (
    df.groupby(["depth", "width"], as_index=False)
      .agg({
          "best_train_acc": "max",
          "best_val_acc": "max",
          "best_test_acc": "max",
          "best_train_loss": "min",
          "best_val_loss": "min",
          "best_test_loss": "min"
      })
      .sort_values(["depth", "width"])
)

# print("\n=== BEST ACCURACY SUMMARY (per width, depth) ===")
# print(summary_df.to_string(index=False))

# Save to CSV
summary_path = f'{folder_name}/{dataset_name}_best_accuracy_summary_{get_timestamp()}.csv'
summary_df.to_csv(summary_path, index=False)



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
import seaborn as sns
from utils.timestamp import get_timestamp
from mup_impl.plot_utils import plot_best_metric, plot_loss
import mup_impl.plot_utils as pu

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
