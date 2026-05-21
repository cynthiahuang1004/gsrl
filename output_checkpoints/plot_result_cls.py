#!/usr/bin/env python3
"""
Training Log Visualizer
Usage: python plot_training_log.py <log_file_path>
"""

import re
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np


def parse_log(log_path):
    """Parse the training log file and extract metrics."""
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_val_acc = []
    lrs = []

    epoch_pattern = re.compile(r'Epoch\s+(\d+)/\d+\s+lr=([\d.e+-]+)')
    result_pattern = re.compile(
        r'→\s+train_loss=([\d.]+)\s+train_acc=([\d.]+)%\s+val_loss=([\d.]+)\s+val_acc=([\d.]+)%'
    )
    best_pattern = re.compile(r'✓ new best val_acc=([\d.]+)%')
    final_best_pattern = re.compile(r'Best val_acc\s*:\s*([\d.]+)%')

    current_epoch = None
    current_lr = None
    best_epochs = set()
    final_best = None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        ep_match = epoch_pattern.search(line)
        if ep_match:
            current_epoch = int(ep_match.group(1))
            current_lr = float(ep_match.group(2))

        res_match = result_pattern.search(line)
        if res_match and current_epoch is not None:
            epochs.append(current_epoch)
            lrs.append(current_lr)
            train_loss.append(float(res_match.group(1)))
            train_acc.append(float(res_match.group(2)))
            val_loss.append(float(res_match.group(3)))
            val_acc.append(float(res_match.group(4)))

        best_match = best_pattern.search(line)
        if best_match and current_epoch is not None:
            best_epochs.add(current_epoch)

        final_match = final_best_pattern.search(line)
        if final_match:
            final_best = float(final_match.group(1))

        i += 1

    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'lrs': lrs,
        'best_epochs': best_epochs,
        'final_best': final_best,
    }


def plot_results(data, log_path, output_path):
    """Generate a styled training visualization chart."""

    epochs = data['epochs']
    train_loss = data['train_loss']
    train_acc = data['train_acc']
    val_loss = data['val_loss']
    val_acc = data['val_acc']
    lrs = data['lrs']
    best_epochs = data['best_epochs']
    final_best = data['final_best']

    # ── Style ──────────────────────────────────────────────────────────────
    BG        = '#0d1117'
    PANEL     = '#161b22'
    BORDER    = '#30363d'
    TRAIN_C   = '#58a6ff'
    VAL_C     = '#f78166'
    LR_C      = '#d2a8ff'
    BEST_C    = '#3fb950'
    TEXT      = '#e6edf3'
    MUTED     = '#8b949e'
    GRID      = '#21262d'

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'text.color': TEXT,
        'axes.facecolor': PANEL,
        'figure.facecolor': BG,
        'axes.edgecolor': BORDER,
        'axes.labelcolor': TEXT,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'lines.linewidth': 2.0,
    })

    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # Title
    sensor_name = os.path.basename(log_path).replace('.log', '')
    fig.suptitle(
        f'Training Dashboard  ·  {sensor_name}',
        fontsize=18, fontweight='bold', color=TEXT,
        y=0.97
    )

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        hspace=0.45, wspace=0.32,
        top=0.91, bottom=0.07, left=0.07, right=0.97
    )

    # ── helper ─────────────────────────────────────────────────────────────
    def style_ax(ax, title, ylabel):
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.set_title(title, color=TEXT, fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('Epoch', color=MUTED, fontsize=9)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(colors=MUTED, labelsize=8)

    def mark_best(ax, data_y):
        for ep in best_epochs:
            if ep in epochs:
                idx = epochs.index(ep)
                ax.axvline(ep, color=BEST_C, linewidth=0.8, linestyle=':', alpha=0.6)
                ax.scatter(ep, data_y[idx], color=BEST_C, s=60, zorder=5,
                           edgecolors='white', linewidths=0.8)

    # ── 1. Loss ────────────────────────────────────────────────────────────
    ax_loss = fig.add_subplot(gs[0, 0])
    style_ax(ax_loss, 'Loss Curve', 'Loss')
    ax_loss.plot(epochs, train_loss, color=TRAIN_C, label='Train Loss', alpha=0.9)
    ax_loss.plot(epochs, val_loss,   color=VAL_C,   label='Val Loss',   alpha=0.9)
    ax_loss.fill_between(epochs, train_loss, alpha=0.08, color=TRAIN_C)
    ax_loss.fill_between(epochs, val_loss,   alpha=0.08, color=VAL_C)
    mark_best(ax_loss, val_loss)
    ax_loss.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

    # ── 2. Accuracy ────────────────────────────────────────────────────────
    ax_acc = fig.add_subplot(gs[0, 1])
    style_ax(ax_acc, 'Accuracy Curve', 'Accuracy (%)')
    ax_acc.plot(epochs, train_acc, color=TRAIN_C, label='Train Acc', alpha=0.9)
    ax_acc.plot(epochs, val_acc,   color=VAL_C,   label='Val Acc',   alpha=0.9)
    ax_acc.fill_between(epochs, train_acc, alpha=0.08, color=TRAIN_C)
    ax_acc.fill_between(epochs, val_acc,   alpha=0.08, color=VAL_C)
    mark_best(ax_acc, val_acc)
    if final_best is not None:
        ax_acc.axhline(final_best, color=BEST_C, linewidth=1.2,
                       linestyle='--', alpha=0.7, label=f'Best {final_best:.2f}%')
    ax_acc.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

    # ── 3. Train/Val Loss (log scale) ──────────────────────────────────────
    ax_log = fig.add_subplot(gs[1, 0])
    style_ax(ax_log, 'Loss Curve (Log Scale)', 'Loss (log)')
    ax_log.semilogy(epochs, train_loss, color=TRAIN_C, label='Train Loss', alpha=0.9)
    ax_log.semilogy(epochs, val_loss,   color=VAL_C,   label='Val Loss',   alpha=0.9)
    ax_log.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

    # ── 4. Overfitting Gap ─────────────────────────────────────────────────
    ax_gap = fig.add_subplot(gs[1, 1])
    style_ax(ax_gap, 'Train / Val Accuracy Gap', 'Gap (%)')
    gap = [t - v for t, v in zip(train_acc, val_acc)]
    ax_gap.plot(epochs, gap, color='#ffa657', alpha=0.9, label='Train−Val Gap')
    ax_gap.fill_between(epochs, gap, alpha=0.12, color='#ffa657')
    ax_gap.axhline(0, color=MUTED, linewidth=0.8, linestyle='-')
    ax_gap.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

    # ── 5. Learning Rate ───────────────────────────────────────────────────
    ax_lr = fig.add_subplot(gs[2, 0])
    style_ax(ax_lr, 'Learning Rate Schedule', 'LR')
    ax_lr.semilogy(epochs, lrs, color=LR_C, alpha=0.9)
    ax_lr.fill_between(epochs, lrs, alpha=0.1, color=LR_C)

    # ── 6. Summary Stats ──────────────────────────────────────────────────
    ax_stats = fig.add_subplot(gs[2, 1])
    ax_stats.set_facecolor(PANEL)
    for spine in ax_stats.spines.values():
        spine.set_edgecolor(BORDER)
    ax_stats.set_xticks([])
    ax_stats.set_yticks([])
    ax_stats.set_title('Summary Statistics', color=TEXT, fontsize=12,
                        fontweight='bold', pad=8)

    best_ep = max(best_epochs) if best_epochs else epochs[-1]
    stats = [
        ('Total Epochs',      f'{len(epochs)}'),
        ('Best Val Acc',      f'{final_best:.2f}%' if final_best else 'N/A'),
        ('Best Epoch',        f'{best_ep}'),
        ('Final Train Acc',   f'{train_acc[-1]:.2f}%'),
        ('Final Val Acc',     f'{val_acc[-1]:.2f}%'),
        ('Min Train Loss',    f'{min(train_loss):.4f}'),
        ('Min Val Loss',      f'{min(val_loss):.4f}'),
        ('Final LR',          f'{lrs[-1]:.2e}'),
    ]

    col_x = [0.05, 0.55]
    row_y = np.linspace(0.82, 0.10, len(stats))
    for i, (k, v) in enumerate(stats):
        ax_stats.text(col_x[0], row_y[i], k, transform=ax_stats.transAxes,
                      color=MUTED, fontsize=9, va='center')
        ax_stats.text(col_x[1], row_y[i], v, transform=ax_stats.transAxes,
                      color=TRAIN_C if 'Train' in k else (BEST_C if 'Best' in k else VAL_C),
                      fontsize=9, va='center', fontweight='bold')

    # Legend for best-epoch markers
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color=TRAIN_C, lw=2, label='Train'),
        Line2D([0], [0], color=VAL_C,   lw=2, label='Validation'),
        Line2D([0], [0], color=BEST_C,  lw=1.5, linestyle=':', label='New Best'),
    ]
    fig.legend(
        handles=legend_els,
        loc='lower center', ncol=3,
        fontsize=9, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT,
        bbox_to_anchor=(0.5, 0.01)
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    print(f'[✓] Saved chart → {output_path}')


def main():
    if len(sys.argv) < 2:
        print('Usage: python plot_training_log.py <log_file_path>')
        sys.exit(1)

    log_path = sys.argv[1]
    if not os.path.exists(log_path):
        print(f'[✗] File not found: {log_path}')
        sys.exit(1)

    output_dir = os.path.dirname(log_path)
    base_name  = os.path.splitext(os.path.basename(log_path))[0]
    output_path = os.path.join(output_dir, f'{base_name}_training_curves.png')

    print(f'[→] Parsing: {log_path}')
    data = parse_log(log_path)
    print(f'[→] Found {len(data["epochs"])} epochs, '
          f'best val_acc={data["final_best"]}%')

    plot_results(data, log_path, output_path)


if __name__ == '__main__':
    main()