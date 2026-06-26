#!/usr/bin/env python
"""Plot DPT / SITR training loss curves from a train_log.csv.

Columns are auto-detected: the Total panel uses train_loss/val_loss, then every
`l_*` component (l_depth, l_normal, l_scl, ...) gets its own panel with the
matching `val_*` overlaid when present. Each panel overlays Train vs Val and
marks the best (min) val epoch. Works for both DPT and SITR logs.

Usage:
    python plot_loss_curve.py <path/to/train_log.csv> [output.png]

If output.png is omitted, writes loss_curve_full.png next to the csv.
"""
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# nice titles for known component keys; falls back to the raw key otherwise
COMPONENT_TITLES = {
    "l_depth": "Depth Loss",
    "l_normal": "Normal Loss",
    "l_scl": "SupCon Loss (l_scl)",
}
COMPONENT_COLORS = ["tab:green", "tab:purple", "tab:brown", "tab:cyan", "tab:olive"]


def load_log(path):
    """Read the csv into a dict of column_name -> list[float]."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty csv: {path}")
    cols = {}
    for key in rows[0].keys():
        vals = []
        for r in rows:
            v = r.get(key, "")
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        cols[key] = vals
    return cols


def plot_pair(ax, epochs, cols, train_key, val_key, title, train_color):
    """One panel: train (solid) + val (dashed) for a given metric pair."""
    has_train = train_key in cols
    has_val = val_key in cols and any(v == v for v in cols[val_key])  # any non-NaN
    if has_train:
        ax.plot(epochs, cols[train_key], color=train_color, label="Train")
    if has_val:
        ax.plot(epochs, cols[val_key], color="red", linestyle="--", label="Val")
        # mark best (min) val epoch
        val = cols[val_key]
        best_i = min(range(len(val)), key=lambda i: val[i] if val[i] == val[i] else float("inf"))
        ax.scatter([epochs[best_i]], [val[best_i]], color="red", zorder=5,
                   s=70, label=f"Best val {val[best_i]:.4f} @ep{int(epochs[best_i])}")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    if has_train or has_val:
        ax.legend(fontsize=8)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    csv_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(os.path.dirname(os.path.abspath(csv_path)), "loss_curve_full.png")

    cols = load_log(csv_path)
    epochs = cols.get("epoch", list(range(len(next(iter(cols.values()))))))

    # discover component losses (any l_* column), in csv order
    components = [k for k in cols if k.startswith("l_")]

    # build the panel list: Total, each component, then LR
    panels = [("Total Loss", "train_loss", "val_loss", "tab:blue")]
    for i, key in enumerate(components):
        title = COMPONENT_TITLES.get(key, key)
        val_key = "val_" + key[2:]  # l_depth -> val_depth
        color = COMPONENT_COLORS[i % len(COMPONENT_COLORS)]
        panels.append((title, key, val_key, color))
    has_lr = "lr" in cols

    n = len(panels) + (1 if has_lr else 0)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows),
                             squeeze=False)
    flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for ax, (title, tkey, vkey, color) in zip(flat, panels):
        plot_pair(ax, epochs, cols, tkey, vkey, title, color)

    idx = len(panels)
    if has_lr:
        ax_lr = flat[idx]
        ax_lr.plot(epochs, cols["lr"], color="tab:orange")
        ax_lr.set_title("Learning Rate")
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_ylabel("lr")
        ax_lr.grid(True, alpha=0.3)
        idx += 1

    # hide any unused axes
    for ax in flat[idx:]:
        ax.axis("off")

    title = os.path.basename(os.path.dirname(os.path.abspath(csv_path))) or "training"
    fig.suptitle(f"Loss Curves — {title}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    print(f"saved: {out_path}")

    # quick text summary
    if "val_loss" in cols:
        val = cols["val_loss"]
        best_i = min(range(len(val)), key=lambda i: val[i] if val[i] == val[i] else float("inf"))
        print(f"best val_loss = {val[best_i]:.4f} @ epoch {int(epochs[best_i])}")
        print(f"final: train={cols['train_loss'][-1]:.4f}  val={val[-1]:.4f}  "
              f"(train/val gap {val[-1] / cols['train_loss'][-1]:.1f}x)")


if __name__ == "__main__":
    main()
