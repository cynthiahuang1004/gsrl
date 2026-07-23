"""
train_dpt_dinov3.py — Train DPT decoder on frozen DINOv3 encoder

Aligned with VisTacFusion:
  - Augmentation: photometric + gel-spin ±180° + center crop 1/√2
  - Depth GT: .npy × 1000
  - Normal GT: rendered PNG / 127.5 - 1.0 → [-1, 1]
  - Input: raw image, ImageNet normalization
  - Loss: MSE + optional Kendall uncertainty weighting
  - Eval: val_depth + val_normal (raw MSE sum) for early stopping

Usage:
  python train_dpt_dinov3.py \
    --dinov3-weights path/to/dinov3_vitl16_pretrain.pth \
    --save-path output_checkpoints/20260709_dpt_dinov3 \
    --device cuda:2 --raw-input --gel-spin-deg 180 --center-crop \
    --depth-from-npy --kendall --lr 2e-4
"""
import argparse
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataloaders import (
    sim_dataset_nested,
    sample_mu, sample_std, norm_mu, norm_std,
    dmap_mu, dmap_std, imagenet_mu, imagenet_std,
)
from models.dpt import DINOv3WithDPT
import torchvision.transforms as T


# ── helpers ─────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count


class WarmupThenPlateau:
    def __init__(self, optimizer, warmup_epochs, warmup_lr, base_lr,
                 factor=0.5, plateau_patience=10, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.group_base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor,
            patience=plateau_patience, min_lr=min_lr)
        self._epoch = 0
        for pg in self.optimizer.param_groups:
            pg["lr"] = warmup_lr * (pg["lr"] / base_lr) if base_lr > 0 else warmup_lr

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def step(self, val_loss=None):
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            frac = self._epoch / max(1, self.warmup_epochs)
            for pg, target in zip(self.optimizer.param_groups, self.group_base_lrs):
                start = self.warmup_lr * (target / self.base_lr) if self.base_lr > 0 else self.warmup_lr
                pg["lr"] = start + (target - start) * frac
        else:
            self.plateau.step(val_loss)

    def state_dict(self):
        return {"epoch": self._epoch, "plateau": self.plateau.state_dict()}
    def load_state_dict(self, d):
        self._epoch = d["epoch"]; self.plateau.load_state_dict(d["plateau"])


def gradient_loss(pred, gt):
    return (F.l1_loss(pred[:, :, :, 1:] - pred[:, :, :, :-1],
                      gt[:, :, :, 1:]  - gt[:, :, :, :-1]) +
            F.l1_loss(pred[:, :, 1:, :] - pred[:, :, :-1, :],
                      gt[:, :, 1:, :]  - gt[:, :, :-1, :]))


# ── train / val ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler,
                    depth_crit, normal_crit, args, device, epoch,
                    log_vars=None):
    model.train()
    loss_m, depth_m, norm_m = AverageMeter(), AverageMeter(), AverageMeter()
    t0 = time.time()
    for step, batch in enumerate(loader):
        imgs     = batch["sample"].to(device, non_blocking=True)
        gt_depth = batch["dmap"].to(device, non_blocking=True)
        gt_norm  = batch["norm"].to(device, non_blocking=True)
        B = imgs.size(0)

        optimizer.zero_grad()
        with autocast("cuda", enabled=args.amp):
            out = model(imgs)
            l_d = depth_crit(out["depth"], gt_depth)
            l_n = normal_crit(out["normal"], gt_norm)

            if log_vars is not None:
                loss = (torch.exp(-log_vars[0]) * l_d + log_vars[0] +
                        torch.exp(-log_vars[1]) * l_n + log_vars[1])
            else:
                loss = args.lambda_depth * l_d + args.lambda_normal * l_n

            if args.lambda_grad > 0:
                loss = loss + args.lambda_grad * gradient_loss(out["depth"], gt_depth)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_m.update(loss.item(), B)
        depth_m.update(l_d.item(), B)
        norm_m.update(l_n.item(), B)

        if step % 50 == 0:
            kd = ""
            if log_vars is not None:
                kd = f"  w_d={torch.exp(-log_vars[0]).item():.2f}  w_n={torch.exp(-log_vars[1]).item():.2f}"
            print(f"  [epoch {epoch:03d} | step {step:04d}/{len(loader):04d}]"
                  f"  loss={loss_m.avg:.4f}  depth={depth_m.avg:.4f}"
                  f"  normal={norm_m.avg:.4f}{kd}  ({time.time()-t0:.1f}s)")
    return loss_m.avg, depth_m.avg, norm_m.avg


@torch.no_grad()
def validate(model, loader, depth_crit, normal_crit, args, device,
             log_vars=None):
    model.eval()
    loss_m, depth_m, norm_m = AverageMeter(), AverageMeter(), AverageMeter()
    for batch in loader:
        imgs     = batch["sample"].to(device, non_blocking=True)
        gt_depth = batch["dmap"].to(device, non_blocking=True)
        gt_norm  = batch["norm"].to(device, non_blocking=True)
        with autocast("cuda", enabled=args.amp):
            out = model(imgs)
            l_d = depth_crit(out["depth"], gt_depth)
            l_n = normal_crit(out["normal"], gt_norm)
            if log_vars is not None:
                loss = (torch.exp(-log_vars[0]) * l_d + log_vars[0] +
                        torch.exp(-log_vars[1]) * l_n + log_vars[1])
            else:
                loss = args.lambda_depth * l_d + args.lambda_normal * l_n
        loss_m.update(loss.item(), imgs.size(0))
        depth_m.update(l_d.item(), imgs.size(0))
        norm_m.update(l_n.item(), imgs.size(0))
    return loss_m.avg, depth_m.avg, norm_m.avg


def plot_loss_curves(history, save_path):
    epochs = history["epochs"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DPT DINOv3 Training Curves", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], color="#2563eb", lw=1.8, label="Train")
    ax.plot(epochs, history["val_loss"],   color="#dc2626", lw=1.8, ls="--", label="Val")
    bi = history["val_loss"].index(min(history["val_loss"]))
    ax.scatter(epochs[bi], history["val_loss"][bi], color="#dc2626", s=80, zorder=5,
               label=f'Best {history["val_loss"][bi]:.4f}')
    ax.set_title("Total Loss"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["l_depth"],  color="#16a34a", lw=1.8)
    axes[0, 1].set_title("Depth Loss"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["l_normal"], color="#9333ea", lw=1.8)
    axes[1, 0].set_title("Normal Loss"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history["lr"],       color="#ea580c", lw=1.8)
    axes[1, 1].set_title("Learning Rate"); axes[1, 1].set_xlabel("Epoch"); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(osp.join(save_path, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="/media/hdd2/ihsuan/gs_blender/renders")
    p.add_argument("--val-every", type=int, default=20)
    p.add_argument("--dinov3-model", default="dinov3_vitl16")
    p.add_argument("--dinov3-weights",
                   default="output_checkpoints/dpt_dinov3/dinov3_vitl16_pretrain_lvd1689m.pth")
    p.add_argument("--dpt-features", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tactile-augment", action="store_true", default=False)
    p.add_argument("--gel-spin-deg", type=float, default=0.0)
    p.add_argument("--center-crop", action="store_true", default=False)
    p.add_argument("--depth-from-npy", action="store_true", default=False)
    p.add_argument("--raw-input", action="store_true", default=False)
    p.add_argument("--lambda-depth",  type=float, default=1.0)
    p.add_argument("--lambda-normal", type=float, default=1.0)
    p.add_argument("--lambda-grad",   type=float, default=0.0)
    p.add_argument("--kendall", action="store_true", default=False)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--plateau-patience", type=int, default=10)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--early-stop", type=int, default=30)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save-path", default="output_checkpoints/dpt_dinov3")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume", default=None)
    p.add_argument("--real-path", default=None,
                   help="Path to real data for sim+real co-training")
    p.add_argument("--real-oversample", type=int, default=0,
                   help="Oversample factor for real data (0=auto)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)

    # ── transforms ──────────────────────────────────────────────────────────
    if args.raw_input:
        img_xform = T.Compose([T.ToTensor(), T.Normalize(mean=imagenet_mu, std=imagenet_std)])
    else:
        img_xform = T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)])
    dmap_xform = T.Compose([T.ToTensor(), T.Normalize(mean=dmap_mu, std=dmap_std)])
    norm_xform = T.Compose([T.ToTensor(), T.Normalize(mean=norm_mu, std=norm_std)])

    # ── dataset (per-session split, matching VisTacFusion) ──────────────────
    print("Loading dataset...")

    def build_ds(augment):
        return sim_dataset_nested(
            path=args.data_path, augment=augment,
            transforms=img_xform, dmap_transforms=dmap_xform, norm_transforms=norm_xform,
            calibration_config=0, sendTwo=False,
            use_gt_norm=True, raw_input=args.raw_input,
            tactile_augment=augment and args.tactile_augment,
            gel_spin_max_deg=args.gel_spin_deg,
            center_crop=args.center_crop,
            depth_from_npy=args.depth_from_npy,
        )

    full_aug   = build_ds(augment=True)
    full_noaug = build_ds(augment=False)
    spu = full_aug.samples_per_unit
    all_idx = list(range(len(full_aug)))
    train_idx = [i for i in all_idx if (i % spu) % args.val_every != 0]
    val_idx   = [i for i in all_idx if (i % spu) % args.val_every == 0]
    train_ds = torch.utils.data.Subset(full_aug, train_idx)
    val_ds   = torch.utils.data.Subset(full_noaug, val_idx)
    print(f"  Per-session split: val_every={args.val_every} "
          f"({len(val_idx)}/{len(all_idx)} val, {len(train_idx)}/{len(all_idx)} train)")

    if args.real_path:
        from torch.utils.data import ConcatDataset
        real_ds = sim_dataset_nested(
            path=args.real_path, augment=False,
            transforms=img_xform, dmap_transforms=dmap_xform, norm_transforms=norm_xform,
            calibration_config=0, sendTwo=False,
            use_gt_norm=False, raw_input=True,
            center_crop=args.center_crop,
            depth_from_npy=True,
        )
        spu_r = real_ds.samples_per_unit
        all_r = list(range(len(real_ds)))
        real_train_idx = [i for i in all_r if (i % spu_r) % args.val_every != 0]
        real_val_idx   = [i for i in all_r if (i % spu_r) % args.val_every == 0]
        real_train = torch.utils.data.Subset(real_ds, real_train_idx)
        real_val   = torch.utils.data.Subset(real_ds, real_val_idx)
        oversample = args.real_oversample
        if oversample <= 0:
            oversample = max(1, len(train_ds) // max(1, len(real_train)) // 2)
        train_ds = ConcatDataset([train_ds] + [real_train] * oversample)
        val_ds = real_val
        print(f"  Co-training: sim + real×{oversample} = {len(train_ds)} train, "
              f"val={len(val_ds)} (real only)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=(args.num_workers > 0),
                              prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=(args.num_workers > 0),
                            prefetch_factor=4 if args.num_workers > 0 else None)
    print(f"  Train: {len(train_ds):,} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_ds):,} samples ({len(val_loader)} batches)")

    # ── model ───────────────────────────────────────────────────────────────
    print(f"\nBuilding DINOv3 encoder ({args.dinov3_model})...")
    model = DINOv3WithDPT(
        model_name=args.dinov3_model, weights=args.dinov3_weights,
        features=args.dpt_features, dropout=args.dropout,
    ).to(device)

    enc_p = sum(p.numel() for p in model.encoder.parameters())
    dec_p = sum(p.numel() for p in model.decoder.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Encoder: {enc_p/1e6:.1f}M (frozen)  Decoder: {dec_p/1e6:.1f}M  Trainable: {trainable/1e6:.1f}M")

    # ── optimizer ───────────────────────────────────────────────────────────
    depth_crit  = nn.MSELoss()
    normal_crit = nn.MSELoss()

    log_vars = None
    if args.kendall:
        log_vars = nn.Parameter(torch.zeros(2, device=device))
        print("  Kendall uncertainty weighting: ON (learned task weights)")

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if log_vars is not None:
        trainable_params.append(log_vars)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = WarmupThenPlateau(
        optimizer, args.warmup_epochs, warmup_lr=1e-6, base_lr=args.lr,
        factor=args.plateau_factor, plateau_patience=args.plateau_patience,
        min_lr=args.min_lr)
    scaler = GradScaler("cuda", enabled=args.amp)

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.decoder.load_state_dict(ck["decoder"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_val_loss = ck.get("best_val_loss", float("inf"))
        if "log_vars" in ck and log_vars is not None:
            log_vars.data = ck["log_vars"]
        print(f"  Resumed from epoch {start_epoch}, best_val={best_val_loss:.4f}")

    # ── training loop ───────────────────────────────────────────────────────
    log_path = osp.join(args.save_path, "train_log.csv")
    history = {"epochs": [], "train_loss": [], "l_depth": [],
               "l_normal": [], "val_loss": [], "lr": []}

    if start_epoch == 0:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,l_depth,l_normal,val_loss,val_depth,val_normal,lr\n")

    epochs_no_improve = 0

    print("\n" + "=" * 60)
    print(f"DPT Training | encoder={args.dinov3_model}  epochs={args.epochs}")
    print(f"  bs={args.batch_size}  lr={args.lr}  dropout={args.dropout}  wd={args.weight_decay}")
    input_mode = "RAW (ImageNet norm)" if args.raw_input else "DIFF (bg-sub)"
    print(f"  input={input_mode}  calib=0  gel_spin={args.gel_spin_deg}°  center_crop={args.center_crop}")
    loss_str = "Kendall (learned)" if args.kendall else f"depth={args.lambda_depth} normal={args.lambda_normal}"
    print(f"  loss: {loss_str}  grad={args.lambda_grad}  depth_from_npy={args.depth_from_npy}")
    print(f"  early_stop={args.early_stop}  device={args.device}")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch:03d}/{args.epochs-1}  lr={lr:.2e}")

        train_loss, l_depth, l_normal = train_one_epoch(
            model, train_loader, optimizer, scaler,
            depth_crit, normal_crit, args, device, epoch, log_vars=log_vars)

        val_loss, val_depth, val_normal = validate(
            model, val_loader, depth_crit, normal_crit, args, device,
            log_vars=log_vars)

        val_metric = val_depth + val_normal
        scheduler.step(val_metric)

        kd_str = ""
        if log_vars is not None:
            w_d = torch.exp(-log_vars[0]).item()
            w_n = torch.exp(-log_vars[1]).item()
            kd_str = f"  (kendall w_depth={w_d:.3f} w_normal={w_n:.3f})"
        print(f"  → train={train_loss:.4f}  depth={l_depth:.4f}  normal={l_normal:.4f}{kd_str}")
        print(f"  → val={val_loss:.4f}  v_depth={val_depth:.4f}  v_normal={val_normal:.4f}  monitor={val_metric:.4f}")

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["l_depth"].append(l_depth)
        history["l_normal"].append(l_normal)
        history["val_loss"].append(val_metric)
        history["lr"].append(lr)

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{l_depth:.6f},{l_normal:.6f},"
                    f"{val_loss:.6f},{val_depth:.6f},{val_normal:.6f},{lr:.2e}\n")

        def save_ckpt(path):
            d = {
                "epoch": epoch,
                "decoder": model.decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "args": vars(args),
            }
            if log_vars is not None:
                d["log_vars"] = log_vars.data
            torch.save(d, path)

        if val_metric < best_val_loss:
            best_val_loss = val_metric
            epochs_no_improve = 0
            save_ckpt(osp.join(args.save_path, "best.pth"))
            print(f"  >>> best val_loss={best_val_loss:.4f} -> best.pth")
        else:
            epochs_no_improve += 1

        if (epoch + 1) % args.save_every == 0:
            save_ckpt(osp.join(args.save_path, f"epoch_{epoch:04d}.pth"))

        save_ckpt(osp.join(args.save_path, "latest.pth"))

        if len(history["epochs"]) >= 2:
            plot_loss_curves(history, args.save_path)

        if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
            print(f"\nEarly stopping: no improvement for {args.early_stop} epochs")
            break

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    if history["epochs"]:
        plot_loss_curves(history, args.save_path)
    print(f"Checkpoints saved -> {args.save_path}/")


if __name__ == "__main__":
    main()
