"""
train_dpt_sitr.py – Stage 2: Train DPT decoder on frozen SITR encoder

Uses simulated tactile data with GT depth maps and surface normals.
The SITR encoder is kept frozen; only the DPT decoder is trained.

Requires: sim data with dmaps/ and norms/ directories per sensor.

Usage
-----
python3 train_dpt_sitr.py \
    --data-path /media/hdd/ihsuan/gsrl/datasets/renders \
    --encoder-weights /media/hdd/ihsuan/gsrl/output_checkpoints/sitr_finetune_v2/sitr_encoder.pth \
    --save-path output_checkpoints/dpt_stage2 \
    --epochs 50 \
    --batch-size 64 \
    --device cuda:0 \
    2>&1 | tee train_dpt_sitr.log
"""

import argparse
import os
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

from dataloaders import (sim_dataset, sim_dataset_nested,
                         sample_mu, sample_std, dmap_mu, dmap_std,
                         norm_mu, norm_std, raw_mu, raw_std,
                         imagenet_mu, imagenet_std)
from models.networks import SITR_base
from models.dpt import SITRWithDPT, DINOv2WithDPT
import torchvision.transforms as T


# ──────────────────────────────────────────────────────────────────────────────
# RAM Cache
# ──────────────────────────────────────────────────────────────────────────────

class CachedDataset(Dataset):
    def __init__(self, dataset, desc="Caching", num_workers=8):
        self.cache = []
        print(f"  {desc}: loading {len(dataset):,} samples into RAM … (workers={num_workers})")
        t0 = time.time()
        if num_workers > 0:
            loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=num_workers, prefetch_factor=2,
                                persistent_workers=False)
            for batch in tqdm(loader, ncols=80):
                self.cache.append({k: v.squeeze(0) for k, v in batch.items()})
        else:
            for i in tqdm(range(len(dataset)), ncols=80):
                self.cache.append(dataset[i])
        print(f"  Done in {time.time() - t0:.1f}s")

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


class WarmupThenPlateau:
    """Warmup LR linearly, then switch to ReduceLROnPlateau.

    Supports multiple param groups with different target LRs.
    Each group warms up proportionally from warmup_lr to its own initial lr.
    """

    def __init__(self, optimizer, warmup_epochs, warmup_lr, base_lr,
                 factor=0.5, plateau_patience=10, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.group_base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor,
            patience=plateau_patience, min_lr=min_lr, verbose=True,
        )
        self._epoch = 0
        for pg in self.optimizer.param_groups:
            pg["lr"] = warmup_lr * (pg["lr"] / base_lr) if base_lr > 0 else warmup_lr

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def step(self, val_loss=None):
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            frac = self._epoch / max(1, self.warmup_epochs)
            for pg, target_lr in zip(self.optimizer.param_groups, self.group_base_lrs):
                start_lr = self.warmup_lr * (target_lr / self.base_lr) if self.base_lr > 0 else self.warmup_lr
                pg["lr"] = start_lr + (target_lr - start_lr) * frac
        else:
            self.plateau.step(val_loss)

    def state_dict(self):
        return {"epoch": self._epoch, "plateau": self.plateau.state_dict()}

    def load_state_dict(self, d):
        self._epoch = d["epoch"]
        self.plateau.load_state_dict(d["plateau"])


def gradient_loss(pred, gt):
    """Spatial gradient loss: encourages correct edges and surface detail."""
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_gt   = gt[:, :, :, 1:]   - gt[:, :, :, :-1]
    dy_gt   = gt[:, :, 1:, :]   - gt[:, :, :-1, :]
    return F.l1_loss(dx_pred, dx_gt) + F.l1_loss(dy_pred, dy_gt)


def plot_loss_curves(history, save_path):
    epochs = history["epochs"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DPT Stage 2 – Loss Curves", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], color="#2563eb", linewidth=1.8, label="Train")
    ax.plot(epochs, history["val_loss"],   color="#dc2626", linewidth=1.8,
            linestyle="--", label="Val")
    best_idx = history["val_loss"].index(min(history["val_loss"]))
    ax.scatter(epochs[best_idx], history["val_loss"][best_idx],
               color="#dc2626", s=80, zorder=5,
               label=f"Best {history['val_loss'][best_idx]:.4f}")
    ax.set_title("Total Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history["l_depth"], color="#16a34a", linewidth=1.8)
    ax.set_title("Depth Loss (MSE)"); ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history["l_normal"], color="#9333ea", linewidth=1.8)
    ax.set_title("Normal Loss (MSE)"); ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, history["lr"], color="#ea580c", linewidth=1.8)
    ax.set_title("Learning Rate"); ax.set_xlabel("Epoch")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(save_path, "loss_curve.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss curve saved → {out_png}")


# ──────────────────────────────────────────────────────────────────────────────
# Train / Validate
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler,
                    depth_criterion, normal_criterion,
                    lambda_depth, lambda_normal, lambda_grad,
                    device, amp_enabled, epoch, log_vars=None):
    model.train()
    loss_m  = AverageMeter()
    depth_m = AverageMeter()
    norm_m  = AverageMeter()
    t0 = time.time()

    for step, batch in enumerate(loader):
        imgs     = batch["sample"].to(device, non_blocking=True)
        calibs   = batch["calibration"].to(device, non_blocking=True)
        gt_depth = batch["dmap"].to(device, non_blocking=True)
        gt_norm  = batch["norm"].to(device, non_blocking=True)
        B = imgs.size(0)

        optimizer.zero_grad()
        with autocast("cuda", enabled=amp_enabled):
            out = model(imgs, calibs)

            l_depth  = depth_criterion(out["depth"], gt_depth)
            l_normal = normal_criterion(out["normal"], gt_norm)

            if log_vars is not None:
                loss = (torch.exp(-log_vars[0]) * l_depth + log_vars[0] +
                        torch.exp(-log_vars[1]) * l_normal + log_vars[1])
            else:
                loss = lambda_depth * l_depth + lambda_normal * l_normal

            if lambda_grad > 0:
                l_grad = gradient_loss(out["depth"], gt_depth)
                loss = loss + lambda_grad * l_grad

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_m.update(loss.item(), B)
        depth_m.update(l_depth.item(), B)
        norm_m.update(l_normal.item(), B)

        if step % 50 == 0:
            elapsed = time.time() - t0
            kd_info = ""
            if log_vars is not None:
                w_d = torch.exp(-log_vars[0]).item()
                w_n = torch.exp(-log_vars[1]).item()
                kd_info = f"  w_d={w_d:.2f}  w_n={w_n:.2f}"
            print(f"  [epoch {epoch:03d} | step {step:04d}/{len(loader):04d}]"
                  f"  loss={loss_m.avg:.4f}"
                  f"  depth={depth_m.avg:.4f}"
                  f"  normal={norm_m.avg:.4f}"
                  f"{kd_info}"
                  f"  ({elapsed:.1f}s)")

    return loss_m.avg, depth_m.avg, norm_m.avg


@torch.no_grad()
def validate(model, loader, depth_criterion, normal_criterion,
             lambda_depth, lambda_normal, device, amp_enabled, log_vars=None):
    model.eval()
    loss_m  = AverageMeter()
    depth_m = AverageMeter()
    norm_m  = AverageMeter()

    for batch in loader:
        imgs     = batch["sample"].to(device, non_blocking=True)
        calibs   = batch["calibration"].to(device, non_blocking=True)
        gt_depth = batch["dmap"].to(device, non_blocking=True)
        gt_norm  = batch["norm"].to(device, non_blocking=True)

        with autocast("cuda", enabled=amp_enabled):
            out = model(imgs, calibs)
            l_depth  = depth_criterion(out["depth"], gt_depth)
            l_normal = normal_criterion(out["normal"], gt_norm)
            if log_vars is not None:
                loss = (torch.exp(-log_vars[0]) * l_depth + log_vars[0] +
                        torch.exp(-log_vars[1]) * l_normal + log_vars[1])
            else:
                loss = lambda_depth * l_depth + lambda_normal * l_normal

        loss_m.update(loss.item(), imgs.size(0))
        depth_m.update(l_depth.item(), imgs.size(0))
        norm_m.update(l_normal.item(), imgs.size(0))

    return loss_m.avg, depth_m.avg, norm_m.avg


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 2: Train DPT decoder on frozen SITR encoder")

    # data
    p.add_argument("--data-path",   type=str, required=True)
    p.add_argument("--layout",      type=str, default="flat", choices=["flat", "nested"],
                   help="nested: path/<obj>/session_xxx/sensor_0000/...（gs_blender 產生的）")
    p.add_argument("--gt-norm",     action="store_true", default=False,
                   help="nested 模式：用 dmaps/xxxx_gt.png 與 norms/xxxx_gt.png（真實幾何）當 Stage 2 監督目標")
    p.add_argument("--val-objects", type=str, nargs="+", default=None,
                   help="nested 模式：指定哪些物體當 validation（held-out by object）；建議與 SITR finetune 用同一組")
    p.add_argument("--val-every",   type=int, default=None,
                   help="Per-session split: every N-th sample is val (e.g. 20 = 5%%)")
    p.add_argument("--val-split",   type=float, default=0.05)
    p.add_argument("--num-sensors", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--no-cache",    action="store_true",
                   help="Disable RAM cache")

    # encoder
    p.add_argument("--encoder",            type=str, default="sitr",
                   choices=["sitr", "dinov2"],
                   help="Encoder backbone: sitr (needs --encoder-weights) or dinov2 (auto-downloads)")
    p.add_argument("--dinov2-model",       type=str, default="dinov2_vitb14",
                   choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                   help="DINOv2 model variant (only used with --encoder dinov2)")
    p.add_argument("--encoder-weights",    type=str, default=None,
                   help="Path to pre-trained SITR encoder weights (required for --encoder sitr)")
    p.add_argument("--calibration-config", type=int, default=18,
                   choices=[0, 4, 8, 9, 18, 19])
    p.add_argument("--raw-input",          action="store_true", default=False,
                   help="Use raw tactile images without background subtraction; auto-sets calib=19")

    # encoder fine-tuning
    p.add_argument("--unfreeze-encoder-layers", type=int, default=0,
                   help="Unfreeze last N transformer blocks of the encoder (0=fully frozen, SITR only)")
    p.add_argument("--encoder-lr", type=float, default=1e-5,
                   help="Learning rate for unfrozen encoder layers (should be << decoder lr)")

    # DPT decoder
    p.add_argument("--dpt-features", type=int, default=256,
                   help="Channel dim inside DPT decoder")
    p.add_argument("--dropout",      type=float, default=0.0,
                   help="Dropout2d rate after fusion (0=off, 0.1-0.3 for regularisation)")

    # augmentation
    p.add_argument("--tactile-augment", action="store_true", default=False,
                   help="Tactile-specific augmentation (photometric: gain/bias/grad/noise)")
    p.add_argument("--gel-spin-deg",    type=float, default=0.0,
                   help="Gel-spin rotation aug max degrees (e.g. 180). 0=off")
    p.add_argument("--center-crop",     action="store_true", default=False,
                   help="Apply fixed 1/sqrt(2) center crop to all samples (train+val)")
    p.add_argument("--depth-from-npy",  action="store_true", default=False,
                   help="Load depth from raw_data/*.npy, compute normal from depth (VisTacFusion convention)")

    # loss weights
    p.add_argument("--lambda-depth",  type=float, default=1.0)
    p.add_argument("--lambda-normal", type=float, default=1.0)
    p.add_argument("--lambda-grad",   type=float, default=0.0,
                   help="Weight for spatial gradient loss on depth (0 = off)")
    p.add_argument("--kendall",       action="store_true", default=False,
                   help="Use Kendall uncertainty weighting (learned task weights, ignores lambda-depth/normal)")

    # training
    p.add_argument("--epochs",        type=int,   default=500,
                   help="Max epochs (early stopping may end sooner)")
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--min-lr",        type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int,   default=5)
    p.add_argument("--weight-decay",  type=float, default=0.05)
    p.add_argument("--scheduler",     type=str, default="plateau",
                   choices=["cosine", "plateau"])
    p.add_argument("--plateau-patience", type=int, default=10)
    p.add_argument("--plateau-factor",   type=float, default=0.5)
    p.add_argument("--early-stop",       type=int, default=30,
                   help="Stop if val loss doesn't improve for N epochs (0=off)")
    p.add_argument("--amp",           action="store_true", default=True)
    p.add_argument("--device",        type=str, default="cuda:0")
    p.add_argument("--num-workers",   type=int, default=4)
    p.add_argument("--no-pin-memory", action="store_true", default=False)
    p.add_argument("--save-path",     type=str, default="checkpoints/dpt_stage2")
    p.add_argument("--save-every",    type=int, default=10)
    p.add_argument("--resume",        type=str, default=None)
    p.add_argument("--seed",          type=int, default=42)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── encoder auto-config ─────────────────────────────────────────────────
    if args.encoder == "sitr" and args.encoder_weights is None:
        raise ValueError("--encoder-weights is required when using --encoder sitr")

    if args.encoder == "dinov2":
        args.raw_input = True
        args.calibration_config = 0
        args.unfreeze_encoder_layers = 0

    if args.raw_input and args.calibration_config not in (0,):
        print(f"  [warn] raw_input=True with calib={args.calibration_config}; "
              f"make sure encoder weights match this config")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.save_path, exist_ok=True)

    # ── dataset (sendTwo=False: single images with GT depth+normal) ──────────
    print("Loading dataset …")

    if args.encoder == "dinov2" or args.raw_input:
        img_xform  = T.Compose([T.ToTensor(), T.Normalize(mean=imagenet_mu, std=imagenet_std)])
    else:
        img_xform  = T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)])
    dmap_xform = T.Compose([T.ToTensor(), T.Normalize(mean=dmap_mu, std=dmap_std)])
    norm_xform = T.Compose([T.ToTensor(), T.Normalize(mean=norm_mu, std=norm_std)])

    def build_dataset(augment, include_objects=None):
        if args.layout == "nested":
            return sim_dataset_nested(
                path=args.data_path,
                augment=augment,
                transforms=img_xform,
                dmap_transforms=dmap_xform,
                norm_transforms=norm_xform,
                tactile_augment=augment and args.tactile_augment,
                calibration_config=args.calibration_config,
                sendTwo=False,
                num_samples=args.num_samples,
                use_gt_norm=args.gt_norm,
                include_objects=include_objects,
                raw_input=args.raw_input,
                gel_spin_max_deg=args.gel_spin_deg,
                center_crop=args.center_crop,
                depth_from_npy=args.depth_from_npy,
            )
        return sim_dataset(
            path=args.data_path,
            augment=augment,
            calibration_config=args.calibration_config,
            sendTwo=False,
            num_samples=args.num_samples,
            num_sensors=args.num_sensors,
        )

    if args.val_every is not None:
        # ── per-session split: every N-th sample is val ──
        full_aug   = build_dataset(augment=True)
        full_noaug = build_dataset(augment=False)
        spu = full_aug.samples_per_unit
        all_idx = list(range(len(full_aug)))
        train_idx = [i for i in all_idx if (i % spu) % args.val_every != 0]
        val_idx   = [i for i in all_idx if (i % spu) % args.val_every == 0]
        train_ds     = torch.utils.data.Subset(full_aug,   train_idx)
        val_ds_noaug = torch.utils.data.Subset(full_noaug, val_idx)
        print(f"Per-session split: val_every={args.val_every} "
              f"({len(val_idx)}/{len(all_idx)} val, "
              f"{len(train_idx)}/{len(all_idx)} train)")

    elif args.layout == "nested" and args.val_objects:
        # ── 按物體切（held-out by object；建議與 SITR finetune 同一組）──
        import glob as _glob
        all_objs = sorted({os.path.basename(os.path.dirname(os.path.dirname(p)))
                           for p in _glob.glob(os.path.join(args.data_path, '*', 'session_*', 'sensor_*'))})
        val_objs = list(args.val_objects)
        missing = [o for o in val_objs if o not in all_objs]
        if missing:
            raise ValueError(f"--val-objects 不存在: {missing}\n可用物體: {all_objs}")
        train_objs = [o for o in all_objs if o not in set(val_objs)]
        print(f"按物體切：train {len(train_objs)} 物體 / val {len(val_objs)} 物體（互不重疊）")
        print(f"  val objects  : {val_objs}")
        print(f"  train objects: {train_objs}")
        train_ds     = build_dataset(augment=True,  include_objects=train_objs)
        val_ds_noaug = build_dataset(augment=False, include_objects=val_objs)
    else:
        full_dataset = build_dataset(augment=True)
        n_val   = max(1, int(len(full_dataset) * args.val_split))
        n_train = len(full_dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        val_ds_noaug = build_dataset(augment=False)
        val_ds_noaug = torch.utils.data.Subset(val_ds_noaug, val_ds.indices)

    # ── workers ────────────────────────────────────────────────────────────
    train_workers = args.num_workers
    val_workers = args.num_workers

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=train_workers, pin_memory=(train_workers > 0 and not args.no_pin_memory),
        drop_last=True,
        persistent_workers=(train_workers > 0),
        prefetch_factor=(4 if train_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds_noaug, batch_size=args.batch_size, shuffle=False,
        num_workers=val_workers, pin_memory=(val_workers > 0 and not args.no_pin_memory),
        persistent_workers=(val_workers > 0),
        prefetch_factor=(4 if val_workers > 0 else None),
    )
    print(f"  train: {len(train_ds):,}   val: {len(val_ds_noaug):,}")

    # sanity check: first sample must have GT depth and normal
    sample0 = train_ds[0]
    assert sample0["dmap"] is not None, \
        "GT depth maps (dmaps/) not found — required for Stage 2"
    assert sample0["norm"] is not None, \
        "GT normals (norms/) not found — required for Stage 2"
    print(f"  GT shapes: dmap={sample0['dmap'].shape}, norm={sample0['norm'].shape}")

    # ── model ─────────────────────────────────────────────────────────────────
    if args.encoder == "dinov2":
        print(f"Building DINOv2 encoder ({args.dinov2_model}) …")
        model = DINOv2WithDPT(
            model_name=args.dinov2_model,
            features=args.dpt_features,
            dropout=args.dropout,
        ).to(device)
    else:
        print("Building SITR encoder …")
        sitr = SITR_base(num_calibration=args.calibration_config)
        state = torch.load(args.encoder_weights, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model_state = sitr.state_dict()
        compatible = {k: v for k, v in state.items()
                      if k in model_state and v.shape == model_state[k].shape}
        skipped = [k for k in state if k not in compatible]
        sitr.load_state_dict(compatible, strict=False)
        if skipped:
            print(f"  Skipped {len(skipped)} keys (shape mismatch): {skipped}")
        print(f"  Loaded {len(compatible)}/{len(state)} encoder keys from {args.encoder_weights}")

        model = SITRWithDPT(
            sitr, embed_dim=768, features=args.dpt_features,
            unfreeze_last_n=args.unfreeze_encoder_layers,
            dropout=args.dropout,
        ).to(device)

    encoder_params  = sum(p.numel() for p in model.encoder.parameters())
    encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    decoder_params  = sum(p.numel() for p in model.decoder.parameters())
    trainable       = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_tag = "frozen" if encoder_trainable == 0 else \
                 f"last {args.unfreeze_encoder_layers} blocks unfrozen, {encoder_trainable/1e6:.1f}M trainable"
    print(f"  Encoder: {encoder_params / 1e6:.1f}M ({frozen_tag})")
    print(f"  Decoder: {decoder_params / 1e6:.1f}M (trainable)")
    print(f"  Total trainable: {trainable / 1e6:.1f}M")

    # ── optimiser ─────────────────────────────────────────────────────────────
    depth_criterion  = nn.MSELoss()
    normal_criterion = nn.MSELoss()

    log_vars = None
    if args.kendall:
        log_vars = nn.Parameter(torch.zeros(2, device=device))
        print("  Kendall uncertainty weighting: ON (learned task weights)")

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if log_vars is not None:
        trainable_params.append(log_vars)

    if args.unfreeze_encoder_layers > 0:
        encoder_params_list = [p for p in model.encoder.parameters() if p.requires_grad]
        decoder_params_list = [p for p in model.decoder.parameters() if p.requires_grad]
        param_groups = [
            {"params": encoder_params_list, "lr": args.encoder_lr},
            {"params": decoder_params_list + ([log_vars] if log_vars is not None else []),
             "lr": args.lr},
        ]
        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.999),
        )
        print(f"  Optimizer: encoder_lr={args.encoder_lr:.1e}, decoder_lr={args.lr:.1e}")
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999),
        )
    if args.scheduler == "cosine":
        scheduler = get_scheduler(
            optimizer, args.warmup_epochs, args.epochs,
            warmup_lr=1e-6, base_lr=args.lr, min_lr=args.min_lr,
        )
    else:
        scheduler = WarmupThenPlateau(
            optimizer, args.warmup_epochs,
            warmup_lr=1e-6, base_lr=args.lr,
            factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            min_lr=args.min_lr,
        )
    scaler = GradScaler("cuda", enabled=args.amp)

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch}")

    # ── history & log ─────────────────────────────────────────────────────────
    history = {
        "epochs": [], "train_loss": [], "l_depth": [],
        "l_normal": [], "val_loss": [], "lr": [],
    }

    log_path = os.path.join(args.save_path, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,l_depth,l_normal,val_loss,val_depth,val_normal,lr\n")

    # ── training loop ─────────────────────────────────────────────────────────
    epochs_no_improve = 0

    print("\n" + "=" * 60)
    enc_name = args.dinov2_model if args.encoder == "dinov2" else "SITR"
    input_mode = "RAW (no bg-sub)" if args.raw_input else "DIFF (bg-sub)"
    print(f"DPT Stage 2  |  encoder={enc_name}  epochs={args.epochs}"
          f"  bs={args.batch_size}  lr={args.lr}  features={args.dpt_features}")
    print(f"Input: {input_mode}  calib={args.calibration_config}")
    if args.unfreeze_encoder_layers > 0:
        print(f"Encoder: last {args.unfreeze_encoder_layers} blocks unfrozen"
              f"  encoder_lr={args.encoder_lr:.1e}")
    else:
        print("Encoder: fully frozen")
    if args.kendall:
        print(f"Loss: Kendall uncertainty weighting (learned)  grad={args.lambda_grad}")
    else:
        print(f"Loss weights: depth={args.lambda_depth}"
              f"  normal={args.lambda_normal}  grad={args.lambda_grad}")
    print(f"Augmentation: {'tactile (gain/bias/grad/noise/flip/rotate)' if args.tactile_augment else 'standard'}")
    print(f"  gel_spin={args.gel_spin_deg}°  center_crop={args.center_crop}  depth_from_npy={args.depth_from_npy}")
    print(f"Regularisation: weight_decay={args.weight_decay}  dropout={args.dropout}")
    print(f"Scheduler: {args.scheduler}  early_stop={args.early_stop}")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch:03d}/{args.epochs - 1}  lr={current_lr:.2e}")

        train_loss, l_depth, l_normal = train_one_epoch(
            model, train_loader, optimizer, scaler,
            depth_criterion, normal_criterion,
            args.lambda_depth, args.lambda_normal, args.lambda_grad,
            device, args.amp, epoch, log_vars=log_vars,
        )

        val_loss, val_depth, val_normal = validate(
            model, val_loader,
            depth_criterion, normal_criterion,
            args.lambda_depth, args.lambda_normal,
            device, args.amp, log_vars=log_vars,
        )
        # Use raw MSE sum for scheduler/early-stop (not Kendall total which drifts negative)
        val_metric = val_depth + val_normal

        if args.scheduler == "plateau":
            scheduler.step(val_metric)
        else:
            scheduler.step()

        kd_str = ""
        if log_vars is not None:
            w_d = torch.exp(-log_vars[0]).item()
            w_n = torch.exp(-log_vars[1]).item()
            kd_str = f"  (kendall w_depth={w_d:.3f} w_normal={w_n:.3f})"
        print(f"  → train={train_loss:.4f}  depth={l_depth:.4f}"
              f"  normal={l_normal:.4f}{kd_str}")
        print(f"  → val={val_loss:.4f}    v_depth={val_depth:.4f}"
              f"  v_normal={val_normal:.4f}  monitor={val_metric:.4f}")

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["l_depth"].append(l_depth)
        history["l_normal"].append(l_normal)
        history["val_loss"].append(val_metric)
        history["lr"].append(current_lr)

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{l_depth:.6f},{l_normal:.6f},"
                    f"{val_loss:.6f},{val_depth:.6f},{val_normal:.6f},"
                    f"{current_lr:.2e}\n")

        # ── checkpoint ────────────────────────────────────────────────────────
        def save_ckpt(path):
            ckpt_dict = {
                "epoch":         epoch,
                "decoder":       model.decoder.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "scaler":        scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "args":          vars(args),
            }
            if log_vars is not None:
                ckpt_dict["log_vars"] = log_vars.data
            if args.unfreeze_encoder_layers > 0 and args.encoder == "sitr":
                ckpt_dict["encoder"] = model.encoder.sitr.state_dict()
            torch.save(ckpt_dict, path)

        if val_metric < best_val_loss:
            best_val_loss = val_metric
            epochs_no_improve = 0
            save_ckpt(os.path.join(args.save_path, "best.pth"))
            print(f"  ✓ best val_loss={best_val_loss:.4f}  →  best.pth")
        else:
            epochs_no_improve += 1

        if (epoch + 1) % args.save_every == 0:
            save_ckpt(os.path.join(args.save_path, f"epoch_{epoch:04d}.pth"))

        save_ckpt(os.path.join(args.save_path, "latest.pth"))

        if len(history["epochs"]) >= 2:
            plot_loss_curves(history, args.save_path)

        if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
            print(f"\n  Early stopping: no improvement for {args.early_stop} epochs")
            break

    # ── final save ────────────────────────────────────────────────────────────
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    plot_loss_curves(history, args.save_path)

    decoder_path = os.path.join(args.save_path, "dpt_decoder.pth")
    torch.save(model.decoder.state_dict(), decoder_path)
    print(f"Decoder saved → {decoder_path}")


if __name__ == "__main__":
    main()
