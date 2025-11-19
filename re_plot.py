from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from utils.timestamp import get_timestamp


# ---------------------------------------------------------------------
# Configure these variables instead of passing command-line arguments.
# ---------------------------------------------------------------------
CSV_PATH = "standard_arxiv_mini_batch/ogbn-arxiv_all_runs_2025-11-08_13-30-49.csv"
OUTPUT_DIR = "re_plots"
DATASET_NAME = "ogbn-arxiv"
LR_CAP = 1e-2


def plot_best_metric(df, metric, title, log_y=False):
    """
    Exact copy of the plotting helper in standard_mini_batch.py, with one addition:
    we optionally filter learning rates using LR_CAP before plotting.
    """
    dff = df.copy()
    dff["lr"] = pd.to_numeric(dff["lr"], errors="coerce")
    dff = dff.dropna(subset=["lr", metric])
    if LR_CAP is not None:
        dff = dff[dff["lr"] <= LR_CAP]

    if dff.empty:
        print(f"[skip] No points remain for {metric} (lr cap={LR_CAP})")
        return

    plt.figure(figsize=(8, 5))
    widths_sorted = sorted(dff["width"].unique())
    depths_sorted = sorted(dff["depth"].unique())
    palette = sns.color_palette("rocket", n_colors=len(depths_sorted))[::-1]
    depth_to_color = {d: palette[i] for i, d in enumerate(depths_sorted)}
    style_cycle = ["solid", "dashed", "dashdot", "dotted", (0, (1, 1))]
    width_to_style = {w: style_cycle[i % len(style_cycle)] for i, w in enumerate(widths_sorted)}

    for (w, d), sub in dff.groupby(["width", "depth"]):
        sub = sub.sort_values("lr")
        x, y = sub["lr"].to_numpy(), sub[metric].to_numpy()
        if log_y:
            mask = y > 0
            x, y = x[mask], y[mask]
            if x.size == 0:
                continue
        plt.plot(
            x,
            y,
            label=f"depth={d}, width={w}",
            color=depth_to_color[d],
            linestyle=width_to_style[w],
            marker="o",
            markersize=4,
        )

    best_idx = dff[metric].idxmax() if "acc" in metric else dff[metric].idxmin()
    best_row = dff.loc[best_idx]
    plt.axhline(best_row[metric], color="gray", linestyle="--", linewidth=1)
    plt.text(
        x=dff["lr"].min(),
        y=best_row[metric] + 0.002,
        s=f"Best {metric}: {best_row[metric]:.4f}",
        color="gray",
        fontsize=10,
        ha="left",
        va="bottom",
    )

    if log_y:
        plt.yscale("log")
    plt.xscale("log")
    lr_label = f"Learning rate (lr ≤ {LR_CAP:g})" if LR_CAP is not None else "Learning rate"
    plt.xlabel(lr_label)
    plt.ylabel(metric.replace("_", " "))
    plt.title(title)

    depth_handles = [
        Line2D([0], [0], color=depth_to_color[d], lw=3, label=str(d)) for d in depths_sorted
    ]
    leg1 = plt.legend(handles=depth_handles, title="Depth", loc="upper right")
    plt.gca().add_artist(leg1)
    if len(widths_sorted) > 1:
        width_handles = [
            Line2D([0], [0], color="black", lw=3, linestyle=width_to_style[w], label=str(w))
            for w in widths_sorted
        ]
        plt.legend(handles=width_handles, title="Width", loc="center right")

    plt.tight_layout()
    safe_title = title.replace(" ", "_")
    suffix = f"_lr_cap_{get_timestamp()}" if LR_CAP is not None else f"_{get_timestamp()}"
    out_path = Path(OUTPUT_DIR) / f"{DATASET_NAME}_{safe_title}{suffix}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[saved] {out_path}")


def main():
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} does not exist")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    available_cols = set(df.columns)

    sns.set(style="whitegrid")
    metrics_to_plot = [
        ("best_train_loss", "Best Train Loss vs LR", True),
        ("best_val_loss", "Best Val Loss vs LR", True),
        ("best_test_loss", "Best Test Loss vs LR", True),
        ("best_train_acc", "Best Train Accuracy vs LR", False),
        ("best_val_acc", "Best Val Accuracy vs LR", False),
        ("best_test_acc", "Best Test Accuracy vs LR", False),
    ]
    for metric, title, log_y in metrics_to_plot:
        plot_best_metric(df, metric, title, log_y=log_y)


if __name__ == "__main__":
    main()
