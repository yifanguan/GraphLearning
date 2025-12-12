import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import seaborn as sns
from utils.timestamp import get_timestamp
import numpy as np

# Global variables to be set by importing code
# passing these variables over and over again is a little bit tedious
folder_name = None
dataset_name = None

def _check():
    if folder_name is None:
        raise ValueError("plot_utils: 'folder_name' is not set.")
    if dataset_name is None:
        raise ValueError("plot_utils: 'dataset_name' is not set.")    

def plot_best_metric(df, metric, title, log_y=False, use_last=False):
    _check()
    if use_last:
        '''
        plot using last epoch result instead of best result
        '''
        if 'best_' in metric:
            raise ValueError("plot_best_metric: best metric is used to plot last metric.")
        metric = f"last_{metric}"

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

    # ---- Find and mark the global best value ----
    if "acc" in metric:
        best_idx = dff[metric].idxmax()   # higher = better
    else:
        best_idx = dff[metric].idxmin()   # lower = better

    # best_idx = dff[metric].idxmax()
    best_row = dff.loc[best_idx]
    best_lr = best_row["lr"]
    best_val = best_row[metric]

    # Add horizontal dashed line and annotation
    plt.axhline(best_val, color="gray", linestyle="--", linewidth=1)
    plt.text(
        x=dff["lr"].min(), y=best_val + 0.002,  # a little above the line
        s=f"Best {metric}: {best_val:.4f}",
        color="gray",
        fontsize=10,
        ha="left",
        va="bottom"
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
    plt.savefig(f'{folder_name}/{dataset_name}_{title}_{get_timestamp()}.png')



def plot_loss(df, lr, title_prefix, log_y=False):
    '''
    For a given learning rate, plot loss vs epoch for all depths
    df: main experiment results table (contains train_loss, val_loss, test_loss columns)
    lr: scalar learning rate to filter rows
    '''
    _check()
    dff = df[df["lr"] == lr].copy()
    if len(dff) == 0:
        print(f"[plot_loss] No data found for lr={lr}")
        return

    # Sort for clean grouping
    dff = dff.sort_values(["width", "depth"])

    # Unique values
    depths_sorted = sorted(dff["depth"].unique())
    widths_sorted = sorted(dff["width"].unique())

    # Color map for different depths
    palette = sns.color_palette("rocket", n_colors=len(depths_sorted))[::-1]
    depth_to_color = {d: palette[i] for i, d in enumerate(depths_sorted)}

    # Linestyles for different widths
    style_cycle = ["solid", "dashed", "dashdot", "dotted", (0, (1, 1))]
    width_to_style = {w: style_cycle[i % len(style_cycle)] for i, w in enumerate(widths_sorted)}

    # -------- helper to plot one split --------
    def plot_split(split_name):
        plt.figure(figsize=(8, 5))

        for (w, d), sub in dff.groupby(["width", "depth"]):
            row = sub.iloc[0]
            curve = row[f"{split_name}_loss"]  # list of (loss, epoch)

            if len(curve) == 0:
                continue

            # unpack tuple list
            epochs = np.array([e for (_, e) in curve])
            losses = np.array([l for (l, _) in curve])

            plt.plot(
                epochs,
                losses,
                color=depth_to_color[d],
                linestyle=width_to_style[w],
                marker="o",
                markersize=4,
                label=f"depth={d}, width={w}"
            )

        # ------- formatting -------
        if log_y:
            plt.yscale("log")

        plt.xlabel("Epoch")
        plt.ylabel(f"{split_name.capitalize()} Loss")
        title = f"{split_name.capitalize()} {title_prefix}"
        plt.title(title)
        # plt.grid(True, linestyle="--", alpha=0.4)

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
        plt.savefig(f"{folder_name}/{dataset_name}_{title}_{get_timestamp()}.png")
        plt.close()

    # ---- make all three plots ----
    plot_split("train")
    plot_split("val")
    plot_split("test")

def save_results(rows, folder_name, dataset_name):
    # record full experiment results
    df = pd.DataFrame(rows).sort_values(["depth", "width", "lr"]).reset_index(drop=True)
    file_path = f"{folder_name}/{dataset_name}_all_runs_{get_timestamp()}.pkl"
    df.to_pickle(file_path)

    print(f"\nSaved all experiment runs to: {file_path}")
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

    # Save Results/Tables
    summary_path = f'{folder_name}/{dataset_name}_best_accuracy_summary_{get_timestamp()}.pkl'
    summary_df.to_pickle(summary_path)
