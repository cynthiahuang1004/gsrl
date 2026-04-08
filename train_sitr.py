"""
train_sitr_new.py – Pre-training script for Sensor-Invariant Tactile Representation (SITR)

Paper : "Sensor-Invariant Tactile Representation", Gupta et al., ICLR 2025
Repo  : https://github.com/hgupt3/gsrl

Key improvement: RAM cache – loads all data into memory once,
then serves from RAM instead of HDD every step.

python3 train_sitr_new.py \
    --data-path /media/hdd/ihsuan/gsrl/datasets/renders \
    --save-path output_checkpoints/sitr_pretrain \
    --device cuda:0
"""

import argparse
import os
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dataloaders import sim_dataset
from models.networks import SITR_base
from models.losses import SupConLoss


# ──────────────────────────────────────────────────────────────────────────────
# RAM Cache Dataset  – loads everything into memory once
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
    p = argparse.ArgumentParser()

    p.add_argument("--data-path",   type=str, required=True)
    p.add_argument("--val-split",   type=float, default=0.05)
    p.add_argument("--num-sensors", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--no-cache",    action="store_true",
                   help="Disable RAM cache (use if dataset too large for RAM)")

    p.add_argument("--calibration-config", type=int, default=18,
                   choices=[0, 4, 8, 9, 18])
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--min-lr",        type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int,   default=5)
    p.add_argument("--weight-decay",  type=float, default=0.05)
    p.add_argument("--lambda-normal", type=float, default=1.0)
    p.add_argument("--lambda-scl",    type=float, default=1.0)
    p.add_argument("--temperature",   type=float, default=0.07)
    p.add_argument("--amp",         action="store_true", default=True)
    p.add_argument("--device",      type=str, default="cuda:0")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-path",   type=str, default="checkpoints/sitr_pretrain")
    p.add_argument("--save-every",  type=int, default=10)
    p.add_argument("--resume",      type=str, default=None)
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

    # ── RAM cache ──────────────────────────────────────────────────────────────
    # Load everything into RAM once → subsequent epochs read from memory, not HDD
    # With 20 sensors (~200K samples), needs ~20-40GB RAM (you have 251GB free)
    # Use --no-cache if RAM is insufficient (e.g. full 1M dataset)
    if not args.no_cache:
        print("Pre-loading data into RAM (one-time cost) …")
        train_ds     = CachedDataset(train_ds,     desc="Train")
        val_ds_noaug = CachedDataset(val_ds_noaug, desc="Val")
        # After caching, num_workers=0 is fastest (no IPC overhead)
        num_workers = 0
    else:
        num_workers = args.num_workers

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(num_workers > 0),
                              drop_last=True)
    val_loader   = DataLoader(val_ds_noaug, batch_size=args.batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=(num_workers > 0))

    print(f"  train: {len(train_ds):,}   val: {len(val_ds_noaug):,}")

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
                               warmup_lr=1e-6, base_lr=args.lr, min_lr=args.min_lr)
    scaler = GradScaler("cuda", enabled=args.amp)

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch}")

    log_path = os.path.join(args.save_path, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,l_normal,l_scl,val_loss,lr\n")

    print("\n" + "=" * 60)
    print(f"SITR Pre-training  |  epochs={args.epochs}  bs={args.batch_size}"
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

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{l_normal:.6f},"
                    f"{l_scl:.6f},{val_loss:.6f},{current_lr:.2e}\n")

        def save_ckpt(path):
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "args": vars(args),
            }, path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(os.path.join(args.save_path, "best.pth"))
            print(f"  ✓ best val_loss={best_val_loss:.4f}  →  best.pth")

        if (epoch + 1) % args.save_every == 0:
            save_ckpt(os.path.join(args.save_path, f"epoch_{epoch:04d}.pth"))

        save_ckpt(os.path.join(args.save_path, "latest.pth"))

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    encoder_path = os.path.join(args.save_path, "sitr_encoder.pth")
    torch.save(model.state_dict(), encoder_path)
    print(f"Encoder saved → {encoder_path}")


if __name__ == "__main__":
    main()