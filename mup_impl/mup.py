# ---------- helpers ----------
# -------- robust fan_in/out + linear accessor --------
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch_geometric.nn import SGConv


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

def mup_param_groups(model, base_lr: float, opt: str = "adam", weight_decay: float = 0.0):
    assert opt in {"sgd", "adam"}
    groups = []

    # Precompute depth scale (number of hidden layers)
    depth = max(1, len(model.fcs))
    # tp6 depth learning rate
    depth_scale = depth ** 0.5  # sqrt(depth)

    # if opt == "adam":
    #     base_lr = base_lr * depth_scale

    # ----- INPUT (SGConv) -----
    fin, fout = fan_in_out(model.sgc)
    W, B = get_linear_like(model.sgc)
    lr_in = base_lr * (fout/fin if opt == "sgd" else 1.0)
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

# ---------- model ----------
class MuGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_fc_layers, K=2, cached=False,
                 activation="gelu", residual_scale=None, bias=False):
        super().__init__()
        self.act = {"relu": F.relu, "gelu": F.gelu, "tanh": torch.tanh}.get(activation, F.relu)

        self.hidden_dim = hidden_dim
        # SGC: linear transform after K-step propagation
        self.sgc = SGConv(in_channels=input_dim, out_channels=hidden_dim, K=K, cached=cached, bias=bias)
        # SGConv contains a .lin (nn.Linear); we μP-init it as "input weights"
        init_mup_input(self.sgc)


        # self.norms = nn.ModuleList([
        #     nn.LayerNorm(hidden_dim) for _ in range(num_fc_layers)
        # ])
        # MLP hidden stack (no bias to keep it clean; add if you want)
        self.fcs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=bias) for _ in range(num_fc_layers)
        ])
        for lin in self.fcs:
            init_mup_hidden(lin)

        # Readout (final linear)
        self.readout = nn.Linear(hidden_dim, output_dim, bias=bias)
    
        # init_mup_readout(self.readout) # option1
        # all zeros initialization tricks
        nn.init.zeros_(self.readout.weight) #option2

        # simple residual scaling like your draft
        # branch multiplier in tp6
        self.multiplier = 3.0
        self.scale = (self.multiplier / math.sqrt(num_fc_layers)) if (num_fc_layers > 0 and residual_scale is None) else (residual_scale or 1.0)

    def forward(self, x, edge_index):
        x = self.sgc(x, edge_index)
        for lin in self.fcs:
            x_in = x
            x = lin(self.act(x))
            x = x_in + self.scale * x
        # x = self.readout(x)
        # x = x / self.hidden_dim # output weight multiplier
        # return x
        return self.readout(x)

def make_mup_optimizer(model, base_lr, opt="adam", weight_decay=0.0, momentum=0.9, betas=(0.9, 0.999)):
    groups = mup_param_groups(model, base_lr, opt, weight_decay)
    if opt == "sgd":
        return torch.optim.SGD(groups, lr=base_lr, momentum=momentum)
    else:
        return torch.optim.Adam(groups, lr=base_lr, betas=betas)
