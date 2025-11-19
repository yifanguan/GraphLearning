import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv


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


# ---------- helpers ----------
# -------- robust fan_in/out + linear accessor --------
import torch.nn as nn

def get_linear_like(module):
    """
    Return (weight_tensor, bias_tensor) for the linear transform inside `module`.
    Works for nn.Linear, PyG convs with .lin, and generic modules exposing .weight.
    """
    # nn.Linear
    if isinstance(module, nn.Linear):
        return module.weight, module.bias

    # PyG convs often expose a .lin (nn.Linear)
    if hasattr(module, "lin"):
        return module.lin.weight, module.lin.bias

    # Generic fallback: try module.weight / module.bias directly
    if hasattr(module, "weight") and getattr(module.weight, "dim", lambda: 0)() == 2:
        bias = module.bias if hasattr(module, "bias") else None
        return module.weight, bias

    raise ValueError(f"Cannot find linear weight in {module.__class__.__name__}")

def fan_in_out(module):
    """
    Compute (fan_in, fan_out) in a way that survives PyG layers and compiled wrappers.
    Priority: explicit attrs -> weight shape.
    """
    # Prefer explicit channel/feature attrs if present
    if hasattr(module, "in_channels") and hasattr(module, "out_channels"):
        return int(module.in_channels), int(module.out_channels)
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        return int(module.in_features), int(module.out_features)

    # Fallback: infer from weight
    W, _ = get_linear_like(module)
    # linear weight is [fan_out, fan_in]
    return int(W.size(1)), int(W.size(0))

def init_mup_input(module):
    # Input weights (& biases): Init Var = 1/fan_in
    fin, _ = fan_in_out(module)
    W, B = get_linear_like(module)
    torch.nn.init.normal_(W, 0.0, 1.0 / math.sqrt(fin))
    if B is not None:
        # Biases use same row as "input weights & all biases": 1/fan_in
        torch.nn.init.normal_(B, 0.0, 1.0 / math.sqrt(fin))

def init_mup_hidden(linear_like):
    # Hidden weights: Init Var = 1/fan_in
    fin, _ = fan_in_out(linear_like)
    W, B = get_linear_like(linear_like)
    torch.nn.init.normal_(W, 0.0, 1.0 / math.sqrt(fin))
    if B is not None:
        # Biases use same row as "input weights & all biases": 1/fan_in
        torch.nn.init.normal_(B, 0.0, 1.0 / math.sqrt(fin))

def init_mup_readout(linear_like):
    # Output (readout) weights: Init Var = 1/fan_in^2
    fin, _ = fan_in_out(linear_like)
    W, B = get_linear_like(linear_like)
    torch.nn.init.normal_(W, 0.0, 1.0 / fin)  # Var = 1/fin^2
    if B is not None:
        # Bias treated like "all biases": 1/fan_in
        torch.nn.init.normal_(B, 0.0, 1.0 / math.sqrt(fin))  # biases follow input/bias rule

# ---------- model ----------
# # class MuGNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_fc_layers, K=2, cached=False,
#                  activation="gelu", residual_scale=None, bias=False):
#         super().__init__()
#         self.act = {"relu": F.relu, "gelu": F.gelu, "tanh": torch.tanh}.get(activation, F.relu)

#         self.hidden_dim = hidden_dim
#         # SGC: linear transform after K-step propagation
#         self.sgc = SGConv(in_channels=input_dim, out_channels=hidden_dim, K=K, cached=cached, bias=bias)
#         # SGConv contains a .lin (nn.Linear); we μP-init it as "input weights"
#         init_mup_input(self.sgc)


#         # self.norms = nn.ModuleList([
#         #     nn.LayerNorm(hidden_dim) for _ in range(num_fc_layers)
#         # ])
#         # MLP hidden stack (no bias to keep it clean; add if you want)
#         self.fcs = nn.ModuleList([
#             nn.Linear(hidden_dim, hidden_dim, bias=bias) for _ in range(num_fc_layers)
#         ])
#         for lin in self.fcs:
#             init_mup_hidden(lin)

#         # Readout (final linear)
#         self.readout = nn.Linear(hidden_dim, output_dim, bias=bias)
#         # init_mup_readout(self.readout)
#         nn.init.zeros_(self.readout.weight)

#         # simple residual scaling like your draft
#         self.scale = (1.0 / math.sqrt(num_fc_layers)) if (num_fc_layers > 0 and residual_scale is None) else (residual_scale or 1.0)

#     def forward(self, x, edge_index):
#         x = self.sgc(x, edge_index)
#         for lin in self.fcs:
#             x_in = x
#             x = lin(self.act(x))
#             x = x_in + self.scale * x
#         # x = self.readout(x)
#         # x = x / self.hidden_dim # output weight multiplier
#         # return x
#         return self.readout(x)

class MuGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_fc_layers, K=2, cached=False,
                 activation="gelu", residual_scale=None, bias=False):
        super().__init__()
        self.act = F.relu
        self.hidden_dim = hidden_dim
        self.sgc = SGConv(in_channels=input_dim, out_channels=hidden_dim, K=K, cached=cached, bias=bias)
        self.fcs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=bias) for _ in range(num_fc_layers)
        ])
        # Readout (final linear)
        self.readout = nn.Linear(hidden_dim, output_dim, bias=bias)

        # simple residual scaling like your draft
        self.scale = 1.0 / math.sqrt(num_fc_layers*hidden_dim)

        # weight init
        set_variance = 1
        ## input embedding
        nn.init.normal_(self.sgc.lin.weight, mean=0.0, std=set_variance)
        
        ## mid layer
        for i in range(len(self.fcs)):
            nn.init.normal_(self.fcs[i].weight, mean=0.0, std=set_variance)
            
        ## output layer
        #nn.init.normal_(self.readout.weight, mean=0.0, std=set_variance)
        nn.init.zeros_(self.readout.weight)


    def forward(self, x, edge_index):
        x = self.sgc(x, edge_index) #input embeding
        for lin in self.fcs:
            # resnet
            x_in = x
            x = lin(self.act(x))
            x = x_in + self.scale * x
        x = self.readout(x)
        x = x/self.hidden_dim # tp setup
        return x  #output embeding


def mup_param_groups(model, base_lr: float, opt: str = "adam", weight_decay: float = 0.0):
    assert opt in {"sgd", "adam"}
    groups = []

    # Precompute depth scale (number of hidden layers)
    depth = max(1, len(model.fcs))
    depth_scale = depth ** 0.5  # sqrt(depth)

    # if opt == "adam":
    #     base_lr = base_lr * depth_scale

    # ----- INPUT (SGConv) -----
    fin, fout = fan_in_out(model.sgc)
    W, B = get_linear_like(model.sgc)
    lr_in = base_lr * (fout if opt == "sgd" else 1.0)
    params = [W] + ([B] if B is not None else [])
    groups.append({"params": params, "lr": lr_in, "weight_decay": weight_decay})

    # ----- HIDDEN -----
    for hlin in model.fcs:
        fin_h, _ = fan_in_out(hlin)
        W_h, B_h = get_linear_like(hlin)
        # weights
        # lr_w = base_lr * (1.0 if opt == "sgd" else (1.0 / fin_h))
        lr_w = base_lr * (1.0 if opt == "sgd" else (1.0 / (fin_h * depth_scale)))
        groups.append({"params": [W_h], "lr": lr_w, "weight_decay": weight_decay})
        # biases follow "input & all biases"
        if B_h is not None:
            fout_h = W_h.size(0)  # infer fan_out from weight shape
            lr_b = base_lr * (fout_h if opt == "sgd" else 1.0)
            groups.append({"params": [B_h], "lr": lr_b, "weight_decay": 0.0})

    # ----- OUTPUT (READOUT) -----
    fin_o, _ = fan_in_out(model.readout)
    W_o, B_o = get_linear_like(model.readout)
    lr_out_w = base_lr / fin_o  # same for SGD & Adam
    groups.append({"params": [W_o], "lr": lr_out_w, "weight_decay": weight_decay})
    if B_o is not None:
        fout_o = W_o.size(0)
        lr_out_b = base_lr * (fout_o if opt == "sgd" else 1.0)
        groups.append({"params": [B_o], "lr": lr_out_b, "weight_decay": 0.0})

    return groups

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
    if dataset_name == 'pubmed' or dataset_name == 'cora' or dataset_name == 'citeseer':
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
widths = [512]
base_width = 256
depths = [1,2,4,8,16]
# depths = [4]
# lrs    = np.linspace(-7, 1, 10)   # add/remove as you like
lrs = [0.1,0.5,1.0,2.0,4.0,8.0]

num_epochs = 5000
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

# for width in widths:
for depth in depths:
    # for log2lr in lrs:
        # lr = 2**log2lr
    for base_lr in lrs:
        key = f"width={base_width} depth={depth} lr={base_lr:g}"
        print(f"\n=== Training {key} ===")

        model = MuGNN(
            input_dim=d, # dataset.num_node_features
            hidden_dim=base_width,
            output_dim=dataset.num_classes, # dataset.num_classes
            num_fc_layers=depth, # depth = num_fc_layers + 1; actually, so we minus one here
            K=2).to(device)

        # optimizer = make_mup_optimizer(model, base_lr=lr, opt="adam", weight_decay=0.0)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr*width, weight_decay=0.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr*base_width)

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
            train_loss = train(model, data, dataset, optimizer)

            if epoch == 1 or epoch % log_every == 0 or epoch == num_epochs:
                m = evaluate(model, data, dataset) # result dict

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
            "width": base_width,
            "depth": depth,
            "lr": base_lr,
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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
import seaborn as sns
from utils.timestamp import get_timestamp


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

def plot_best_metric(df, metric, title, log_y=False):
    # Clean up
    dff = df.copy()
    dff["lr"] = pd.to_numeric(dff["lr"], errors="coerce")
    dff = dff.dropna(subset=["lr", metric])

    plt.figure(figsize=(8, 5))

    widths_sorted = sorted(dff["width"].unique())
    depths_sorted = sorted(dff["depth"].unique())

    # Depth → color (rocket reversed: black→orange)
    palette = sns.color_palette("rocket", n_colors=len(depths_sorted))[::-1]
    depth_to_color = {d: palette[i] for i, d in enumerate(depths_sorted)}

    # Width → linestyle
    style_cycle = ["solid", "dashed", "dashdot", "dotted", (0, (1, 1))]
    width_to_style = {w: style_cycle[i % len(style_cycle)] for i, w in enumerate(widths_sorted)}

    # Plot one curve per (width, depth)
    for (w, d), sub in dff.groupby(["width", "depth"]):
        sub = sub.sort_values("lr")
        x = sub["lr"].to_numpy()
        y = sub[metric].to_numpy()

        if log_y:
            mask = y > 0
            x, y = x[mask], y[mask]
            if x.size == 0:
                continue

        plt.plot(
            x, y,
            label=f"depth={d}, width={w}",
            color=depth_to_color[d],      # depth controls color
            linestyle=width_to_style[w],  # width controls line shape
            marker="o",
            markersize=4,
        )

    # Scales and labels
    if log_y:
        plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel(metric.replace("_", " "))
    plt.title(title)

    # ----- Legends -----
    depth_handles = [Line2D([0],[0], color=depth_to_color[d], lw=3, label=str(d))
                     for d in depths_sorted]
    leg1 = plt.legend(handles=depth_handles,
                      title="Depth",
                      loc="upper right")
    plt.gca().add_artist(leg1)

    # Add width legend only if multiple widths exist
    if len(widths_sorted) > 1:
        width_handles = [Line2D([0],[0], color="black", lw=3,
                                linestyle=width_to_style[w], label=str(w))
                         for w in widths_sorted]
        plt.legend(handles=width_handles, title="Width", loc="center right")

    plt.tight_layout()
    plt.savefig(f'mup_tintin_repro/{dataset_name}_{title}_{get_timestamp()}.png')


plot_best_metric(df, "best_train_loss", "Best Train Loss vs LR", log_y=True)
plot_best_metric(df, "best_val_loss",  "Best Val Loss vs LR",   log_y=True)
plot_best_metric(df, "best_test_loss", "Best Test Loss vs LR", log_y=True)

plot_best_metric(df, "best_train_acc", "Best Train Accuracy vs LR", log_y=False)
plot_best_metric(df, "best_val_acc",  "Best Val Accuracy vs LR",   log_y=False)
plot_best_metric(df, "best_test_acc", "Best Test Accuracy vs LR", log_y=False)
