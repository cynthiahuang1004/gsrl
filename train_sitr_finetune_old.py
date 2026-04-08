"""
train_sitr_finetune.py: Continued pre-training from existing SITR weights
 
Paper : "Sensor-Invariant Tactile Representation", Gupta et al., ICLR 2025
Repo  : https://github.com/hgupt3/gsrl

改動重點 (相較於 train_sitr.py)
  1. 新增 --pretrain-weights: 載入 weights-only checkpoint (SITR_B18.pth)
  2. --lr 預設改為 1e-5 (原本 3e-4)，適合 fine-tuning
  3. --warmup-epochs 預設改為 2 (原本 5)
  4. --epochs 預設改為 20 (原本 100)
  5. 其餘 (RAM cache、training loop、logging) 與 train_sitr.py 完全相同
  6. 訓練結束後自動輸出 loss_curve.png

Usage
-----
python3 train_sitr_finetune.py \
    --data-path /media/hdd/ihsuan/gsrl/datasets/renders_lmdb \
    --pretrain-weights /media/hdd/ihsuan/gsrl/datasets/checkpoints/SITR_B18.pth \
    --save-path output_checkpoints/sitr_finetune \
    --lr 1e-5 \
    --epochs 20 \
    --device cuda:3 \
    --num-workers 8 
"""

import argparse
import os
import time
import math
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from tqdm import tqdm
# from dataloaders import sim_dataset
from lmdb_dataset import LMDBDataset
from models.networks import SITR_base
from models.losses import SupConLoss


# ──────────────────────────────────────────────────────────────────────────────
# RAM Cache Dataset
# ──────────────────────────────────────────────────────────────────────────────

class CachedDataset(Dataset):
    """
    Wraps any map-style dataset and pre-loads all samples into RAM.
    After loading, __getitem__ is just a list lookup – no HDD access.
    """
    def __init__(self, dataset, desc="Caching"):
        self.cache = []
        print(f"  {desc}: loading {len(dataset):,} samples into RAM …")
        t0 = time.time()
        for i in tqdm(range(len(dataset)), ncols=80):
            self.cache.append(dataset[i])
        print(f"  Done in {time.time()-t0:.1f}s")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def get_scheduler(optimizer, warmup_epochs, total_epochs,
                  warmup_lr, base_lr, min_lr=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return warmup_lr / base_lr + \
                   (1.0 - warmup_lr / base_lr) * epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr + (base_lr - min_lr) * cosine) / base_lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# Loss Curve Plotting
# ──────────────────────────────────────────────────────────────────────────────
 
def plot_loss_curves(history: dict, save_path: str, start_epoch: int = 0):
    """
    Draws and saves a loss curve PNG from training history.
 
    Parameters
    ----------
    history : dict with keys:
        "epochs"      : list[int]
        "train_loss"  : list[float]
        "l_normal"    : list[float]
        "l_scl"       : list[float]
        "val_loss"    : list[float]
        "lr"          : list[float]
    save_path : str   – output directory
    start_epoch : int – used only for axis labelling
    """
    epochs     = history["epochs"]
    train_loss = history["train_loss"]
    l_normal   = history["l_normal"]
    l_scl      = history["l_scl"]
    val_loss   = history["val_loss"]
    lr_vals    = history["lr"]
 
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("SITR Fine-tuning – Loss Curves", fontsize=14, fontweight="bold")
 
    # ── subplot 1: Total Loss ─────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, color="#2563eb", linewidth=1.8, label="Train Loss")
    ax.plot(epochs, val_loss,   color="#dc2626", linewidth=1.8,
            linestyle="--", label="Val Loss")
    # mark best val
    best_idx = val_loss.index(min(val_loss))
    ax.scatter(epochs[best_idx], val_loss[best_idx],
               color="#dc2626", s=80, zorder=5,
               label=f"Best Val {val_loss[best_idx]:.4f}")
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
 
    # ── subplot 2: Normal Loss ────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, l_normal, color="#16a34a", linewidth=1.8)
    ax.set_title("Normal Loss (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
 
    # ── subplot 3: Contrastive Loss ───────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(epochs, l_scl, color="#9333ea", linewidth=1.8)
    ax.set_title("Supervised Contrastive Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
 
    # ── subplot 4: Learning Rate ──────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(epochs, lr_vals, color="#ea580c", linewidth=1.8)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    out_png = os.path.join(save_path, "loss_curve.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss curve saved → {out_png}")


# ──────────────────────────────────────────────────────────────────────────────
# Train one epoch
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, normal_criterion,
                    contrastive_criterion, lambda_normal, lambda_scl,
                    device, amp_enabled, epoch):
    model.train()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scl_meter  = AverageMeter()
    t0 = time.time()

    for step, batch in enumerate(loader):
        imgs   = batch["sample"].to(device, non_blocking=True)
        calibs = batch["calibration"].to(device, non_blocking=True)
        norms  = batch["norm"].to(device, non_blocking=True)
        labels = batch["idx"].to(device, non_blocking=True)

        img_a,   img_b   = imgs[:, 0],   imgs[:, 1]
        calib_a, calib_b = calibs[:, 0], calibs[:, 1]
        norm_a,  norm_b  = norms[:, 0],  norms[:, 1]
        B = img_a.size(0)

        optimizer.zero_grad()

        with autocast("cuda", enabled=amp_enabled):
            out_a = model(img_a, calib_a)
            out_b = model(img_b, calib_b)

            l_normal = (normal_criterion(out_a["proj"], norm_a) +
                        normal_criterion(out_b["proj"], norm_b)) * 0.5

            feats = torch.stack([out_a["cls_token"],
                                 out_b["cls_token"]], dim=1)  # (B, 2, 128)
            l_scl = contrastive_criterion(feats, labels)

            loss = lambda_normal * l_normal + lambda_scl * l_scl

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(),     B)
        norm_meter.update(l_normal.item(), B)
        scl_meter.update(l_scl.item(),     B)

        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [epoch {epoch:03d} | step {step:04d}/{len(loader):04d}]"
                  f"  loss={loss_meter.avg:.4f}"
                  f"  l_norm={norm_meter.avg:.4f}"
                  f"  l_scl={scl_meter.avg:.4f}"
                  f"  ({elapsed:.1f}s)")

    return loss_meter.avg, norm_meter.avg, scl_meter.avg


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, normal_criterion, device, amp_enabled):
    model.eval()
    meter = AverageMeter()
    for batch in loader:
        img_a   = batch["sample"][:, 0].to(device, non_blocking=True)
        calib_a = batch["calibration"][:, 0].to(device, non_blocking=True)
        norm_a  = batch["norm"][:, 0].to(device, non_blocking=True)

        with autocast("cuda", enabled=amp_enabled):
            out  = model(img_a, calib_a)
            loss = normal_criterion(out["proj"], norm_a)
        meter.update(loss.item(), img_a.size(0))
    return meter.avg


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SITR continued pre-training from existing weights (e.g. SITR_B18.pth)")
 
    p.add_argument("--data-path",   type=str, required=True)
    p.add_argument("--val-split",   type=float, default=0.05)
    p.add_argument("--num-sensors", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--no-cache",    action="store_true",
                   help="Disable RAM cache (use if dataset too large for RAM)")
 
    p.add_argument("--pretrain-weights", type=str, default=None,
                   help="Path to weights-only checkpoint (e.g. SITR_B18.pth). "
                        "載入 model weights，optimizer/scheduler 重新初始化。"
                        "與 --resume 擇一使用，--resume 優先。")
 
    p.add_argument("--calibration-config", type=int, default=18,
                   choices=[0, 4, 8, 9, 18])
 
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=1e-5)
    p.add_argument("--min-lr",        type=float, default=1e-7)
    p.add_argument("--warmup-epochs", type=int,   default=2)
    p.add_argument("--weight-decay",  type=float, default=0.05)
    p.add_argument("--lambda-normal", type=float, default=1.0)
    p.add_argument("--lambda-scl",    type=float, default=1.0)
    p.add_argument("--temperature",   type=float, default=0.07)
    p.add_argument("--amp",         action="store_true", default=True)
    p.add_argument("--device",      type=str, default="cuda:0")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-path",   type=str, default="checkpoints/sitr_finetune")
    p.add_argument("--save-every",  type=int, default=5)
    p.add_argument("--resume",      type=str, default=None,
                   help="完整 checkpoint（含 optimizer/scheduler），優先於 --pretrain-weights")
    p.add_argument("--seed",        type=int, default=42)    
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.save_path, exist_ok=True)

    # ── datasets ──────────────────────────────────────────────────────────────
    '''
    print("Loading dataset …")
    full_dataset = sim_dataset(
        path=args.data_path,
        augment=True,
        calibration_config=args.calibration_config,
        sendTwo=True,
        num_samples=args.num_samples,
        num_sensors=args.num_sensors,
    )

    n_val   = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    val_ds_noaug = sim_dataset(
        path=args.data_path,
        augment=False,
        calibration_config=args.calibration_config,
        sendTwo=True,
        num_samples=args.num_samples,
        num_sensors=args.num_sensors,
    )
    val_ds_noaug = torch.utils.data.Subset(val_ds_noaug, val_ds.indices)

    # ── RAM cache ─────────────────────────────────────────────────────────────
    if not args.no_cache:
        print("Pre-loading data into RAM (one-time cost) …")
        train_ds     = CachedDataset(train_ds,     desc="Train")
        val_ds_noaug = CachedDataset(val_ds_noaug, desc="Val")
        num_workers = 0
    else:
        num_workers = args.num_workers

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(num_workers > 0),
                              drop_last=True, prefetch_factor=4)
    val_loader   = DataLoader(val_ds_noaug, batch_size=args.batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=(num_workers > 0))
    
    print(f"  train: {len(train_ds):,}   val: {len(val_ds_noaug):,}")
    '''
    
    # ── datasets (LMDB) ───────────────────────────────────────────────────────
    print("Loading dataset (LMDB) …")
    full_train = LMDBDataset(args.data_path, augment=True,  sendTwo=True,
                             calibration_config=args.calibration_config)
    full_val   = LMDBDataset(args.data_path, augment=False, sendTwo=True,
                             calibration_config=args.calibration_config)

    n_val   = max(1, int(len(full_train) * args.val_split))
    n_train = len(full_train) - n_val

    rng     = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(full_train), generator=rng).tolist()
    train_indices = indices[:n_train]
    val_indices   = indices[n_train:]

    train_ds     = torch.utils.data.Subset(full_train, train_indices)
    val_ds_noaug = torch.utils.data.Subset(full_val,   val_indices)

    print(f"  train: {len(train_ds):,}   val: {len(val_ds_noaug):,}")

    num_workers = args.num_workers  # LMDB 多讀者安全，直接用
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=(num_workers > 0),
                              prefetch_factor=(4 if num_workers > 0 else None))
    val_loader   = DataLoader(val_ds_noaug, batch_size=args.batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True,
                              persistent_workers=(num_workers > 0),
                              prefetch_factor=(2 if num_workers > 0 else None))
    
    # ── model ─────────────────────────────────────────────────────────────────
    print("Building model …")
    model = SITR_base(num_calibration=args.calibration_config).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    normal_criterion      = nn.MSELoss()
    contrastive_criterion = SupConLoss(temperature=args.temperature)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999))
    scheduler = get_scheduler(optimizer, args.warmup_epochs, args.epochs,
                               warmup_lr=1e-8, base_lr=args.lr, min_lr=args.min_lr)
    scaler = GradScaler("cuda", enabled=args.amp)

    # ── weights loading（優先順序：--resume > --pretrain-weights） ────────────
    start_epoch   = 0
    best_val_loss = float("inf")

    if args.resume is not None:
        print(f"Resuming from full checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    elif args.pretrain_weights is not None:
        # ── [核心改動] weights-only checkpoint（如 SITR_B18.pth） ────────────
        # optimizer/scheduler 重新初始化，從 epoch 0 開始，lr 用 args.lr
        print(f"Loading pretrained weights (weights-only): {args.pretrain_weights}")
        state = torch.load(args.pretrain_weights, map_location=device)
        model.load_state_dict(state)
        print(f"  Loaded. Optimizer & scheduler re-initialized with lr={args.lr:.1e}")

    else:
        print("  WARNING: No weights provided – training from random init.")

    # ── history dict：蒐集每 epoch 的指標 ─────────────────────────────
    history = {
        "epochs":     [],
        "train_loss": [],
        "l_normal":   [],
        "l_scl":      [],
        "val_loss":   [],
        "lr":         [],
    }

    # ── log ───────────────────────────────────────────────────────────────────
    log_path = os.path.join(args.save_path, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,l_normal,l_scl,val_loss,lr\n")

    # ── training loop ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"SITR Fine-tuning  |  epochs={args.epochs}  bs={args.batch_size}"
          f"  lr={args.lr}  τ={args.temperature}")
    print(f"RAM cache: {'OFF (--no-cache)' if args.no_cache else 'ON'}")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch:03d}/{args.epochs-1}  lr={current_lr:.2e}")

        train_loss, l_normal, l_scl = train_one_epoch(
            model, train_loader, optimizer, scaler,
            normal_criterion, contrastive_criterion,
            args.lambda_normal, args.lambda_scl,
            device, args.amp, epoch)

        val_loss = validate(model, val_loader, normal_criterion, device, args.amp)
        scheduler.step()

        print(f"  → train={train_loss:.4f}  l_norm={l_normal:.4f}"
              f"  l_scl={l_scl:.4f}  val={val_loss:.4f}")

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["l_normal"].append(l_normal)
        history["l_scl"].append(l_scl)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{l_normal:.6f},"
                    f"{l_scl:.6f},{val_loss:.6f},{current_lr:.2e}\n")

        def save_ckpt(path):
            torch.save({
                "epoch":          epoch,
                "model":          model.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "scheduler":      scheduler.state_dict(),
                "scaler":         scaler.state_dict(),
                "best_val_loss":  best_val_loss,
                "args":           vars(args),
            }, path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(os.path.join(args.save_path, "best.pth"))
            print(f"  ✓ best val_loss={best_val_loss:.4f}  →  best.pth")

        if (epoch + 1) % args.save_every == 0:
            save_ckpt(os.path.join(args.save_path, f"epoch_{epoch:04d}.pth"))

        save_ckpt(os.path.join(args.save_path, "latest.pth"))

        if len(history["epochs"]) >= 2:   
            plot_loss_curves(history, args.save_path, start_epoch=start_epoch)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    plot_loss_curves(history, args.save_path, start_epoch=start_epoch)
    encoder_path = os.path.join(args.save_path, "sitr_encoder.pth")
    torch.save(model.state_dict(), encoder_path)
    print(f"Encoder saved → {encoder_path}")


if __name__ == "__main__":
    main()