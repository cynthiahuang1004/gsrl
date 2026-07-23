"""
train_pose_dinov3.py — Train pose estimation head on frozen DINOv3 encoder

Aligned with VisTacFusion:
  - Augmentation: photometric + gel-spin ±180° + center crop 1/√2
  - Input: raw image, ImageNet normalization, no calibration
  - Pose: SE(2) regression (1-cos + L1), matching VisTacFusion convention
  - Loss: optional Kendall uncertainty weighting

Usage:
  python train_pose_dinov3.py \
    --dinov3-weights path/to/dinov3_vitl16_pretrain.pth \
    --save-path output_checkpoints/20260709_pose_dinov3 \
    --device cuda:3 --raw-input --gel-spin-deg 180 --center-crop --kendall
"""
import argparse
import math
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataloaders import imagenet_mu, imagenet_std, sample_mu, sample_std
from models.dpt import DINOv3MultiScale, _convert_dinov3_hf_to_hub, auto_layer_indices, _count_blocks
from train_pose_sitr import PoseHead, PoseLoss, PoseDataset, AverageMeter
import torchvision.transforms as T


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


class DINOv3Encoder(nn.Module):
    """Frozen DINOv3 encoder that returns cls_token + patch tokens for pose head."""

    def __init__(self, model_name='dinov3_vitl16', weights=None):
        super().__init__()
        self.dinov3 = torch.hub.load(
            'facebookresearch/dinov3', model_name, pretrained=False)

        print(f'  [encoder] loading weights from {weights}')
        state_dict = torch.load(weights, map_location='cpu', weights_only=True)
        state_dict = _convert_dinov3_hf_to_hub(state_dict, self.dinov3)
        missing, unexpected = self.dinov3.load_state_dict(state_dict, strict=False)
        num_buf_missing = sum(1 for k in missing
                              if k in dict(self.dinov3.named_buffers()))
        num_param_missing = len(missing) - num_buf_missing
        if num_param_missing:
            print(f'  [encoder] WARNING: {num_param_missing} parameter keys NOT loaded')
        print(f'  [encoder] loaded {len(state_dict) - len(unexpected)} / '
              f'{sum(1 for _ in self.dinov3.parameters())} parameters')

        self.embed_dim = getattr(self.dinov3, 'embed_dim', None)
        if self.embed_dim is None:
            self.embed_dim = self.dinov3.norm.normalized_shape[0]

        for p in self.dinov3.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.dinov3.eval()
        return self

    @torch.no_grad()
    def forward(self, x, c=None):
        out = self.dinov3.forward_features(x)
        cls_token = out["x_norm_clstoken"].unsqueeze(1)  # (B, 1, D)
        patch_tokens = out["x_norm_patchtokens"]          # (B, N, D)
        latent = torch.cat([cls_token, patch_tokens], dim=1)
        return {"latent": latent}


def plot_loss_curves(history, save_path):
    epochs = history["epochs"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Pose DINOv3 Training Curves", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], color="#2563eb", lw=1.8, label="Train")
    ax.plot(epochs, history["val_loss"],   color="#dc2626", lw=1.8, ls="--", label="Val")
    bi = history["val_loss"].index(min(history["val_loss"]))
    ax.scatter(epochs[bi], history["val_loss"][bi], color="#dc2626", s=80, zorder=5,
               label=f'Best {history["val_loss"][bi]:.4f}')
    ax.set_title("Total Loss"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["l_rot"],   color="#16a34a", lw=1.8)
    axes[0, 1].set_title("Rotation Loss"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["l_trans"], color="#9333ea", lw=1.8)
    axes[1, 0].set_title("Translation Loss"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history["lr"],      color="#ea580c", lw=1.8)
    axes[1, 1].set_title("Learning Rate"); axes[1, 1].set_xlabel("Epoch"); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(osp.join(save_path, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Train / Val (same structure as train_pose_sitr.py) ─────────────────────

def train_one_epoch(encoder, pose_head, loader, optimizer, scaler,
                    pose_loss_fn, w_rot, w_trans, device, amp_enabled, epoch,
                    log_vars=None, obj_embedding=None):
    pose_head.train()
    loss_m, rot_m, trans_m = AverageMeter(), AverageMeter(), AverageMeter()
    t0 = time.time()

    for step, batch in enumerate(loader):
        imgs    = batch["sample"].to(device, non_blocking=True)
        calibs  = batch["calibration"].to(device, non_blocking=True)
        gt_pose = batch["pose"].to(device, non_blocking=True)
        B = imgs.size(0)

        optimizer.zero_grad()
        with autocast("cuda", enabled=amp_enabled):
            with torch.no_grad():
                enc_out = encoder(imgs, calibs)
            latent = enc_out["latent"]
            if obj_embedding is not None and "object" in batch:
                obj_ids = batch["object"].to(device, non_blocking=True)
                obj_emb = obj_embedding(obj_ids).unsqueeze(1)
                latent = latent + obj_emb
            cls_token = latent[:, 0, :]
            spatial = latent[:, 1:, :]
            pred = pose_head(cls_token, spatial)
            l_rot, l_trans = pose_loss_fn(pred, gt_pose)

            if log_vars is not None:
                loss = (torch.exp(-log_vars[0]) * l_rot + 0.5 * log_vars[0] +
                        torch.exp(-log_vars[1]) * l_trans + 0.5 * log_vars[1])
            else:
                loss = w_rot * l_rot + w_trans * l_trans

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(pose_head.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_m.update(loss.item(), B)
        rot_m.update(l_rot.item(), B)
        trans_m.update(l_trans.item(), B)

        if step % 50 == 0:
            kd = ""
            if log_vars is not None:
                kd = f"  w_r={torch.exp(-log_vars[0]).item():.2f}  w_t={torch.exp(-log_vars[1]).item():.2f}"
            print(f"  [epoch {epoch:03d} | step {step:04d}/{len(loader):04d}]"
                  f"  loss={loss_m.avg:.4f}  rot={rot_m.avg:.4f}  trans={trans_m.avg:.4f}"
                  f"{kd}  ({time.time()-t0:.1f}s)")

    return loss_m.avg, rot_m.avg, trans_m.avg


@torch.no_grad()
def validate(encoder, pose_head, loader, pose_loss_fn, w_rot, w_trans,
             device, amp_enabled, log_vars=None, obj_embedding=None):
    pose_head.eval()
    loss_m, rot_m, trans_m = AverageMeter(), AverageMeter(), AverageMeter()
    all_se2_pred, all_se2_gt = [], []

    for batch in loader:
        imgs    = batch["sample"].to(device, non_blocking=True)
        calibs  = batch["calibration"].to(device, non_blocking=True)
        gt_pose = batch["pose"].to(device, non_blocking=True)

        with autocast("cuda", enabled=amp_enabled):
            enc_out = encoder(imgs, calibs)
            latent = enc_out["latent"]
            if obj_embedding is not None and "object" in batch:
                obj_ids = batch["object"].to(device, non_blocking=True)
                obj_emb = obj_embedding(obj_ids).unsqueeze(1)
                latent = latent + obj_emb
            cls_token = latent[:, 0, :]
            spatial = latent[:, 1:, :]
            pred = pose_head(cls_token, spatial)
            l_rot, l_trans = pose_loss_fn(pred, gt_pose)
            if log_vars is not None:
                loss = (torch.exp(-log_vars[0]) * l_rot + 0.5 * log_vars[0] +
                        torch.exp(-log_vars[1]) * l_trans + 0.5 * log_vars[1])
            else:
                loss = w_rot * l_rot + w_trans * l_trans

        loss_m.update(loss.item(), imgs.size(0))
        rot_m.update(l_rot.item(), imgs.size(0))
        trans_m.update(l_trans.item(), imgs.size(0))
        all_se2_pred.append(pred["se2"].float().cpu())
        all_se2_gt.append(gt_pose.cpu())

    se2_pred = torch.cat(all_se2_pred)
    se2_gt = torch.cat(all_se2_gt)
    theta_pred = torch.atan2(se2_pred[:, 1], se2_pred[:, 0])
    theta_gt = torch.atan2(se2_gt[:, 1], se2_gt[:, 0])
    rot_err_deg = torch.abs(theta_pred - theta_gt)
    rot_err_deg = torch.min(rot_err_deg, 2 * math.pi - rot_err_deg) * 180 / math.pi
    trans_err = (se2_pred[:, 2:] - se2_gt[:, 2:]).abs().mean().item()

    return loss_m.avg, rot_m.avg, trans_m.avg, rot_err_deg.mean().item(), trans_err


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="/media/hdd2/ihsuan/gs_blender/renders")
    p.add_argument("--mesh-dir",  default="/media/hdd2/ihsuan/gs_blender/meshes")
    p.add_argument("--dinov3-model", default="dinov3_vitl16")
    p.add_argument("--dinov3-weights",
                   default="output_checkpoints/dpt_dinov3/dinov3_vitl16_pretrain_lvd1689m.pth")
    p.add_argument("--val-every", type=int, default=20)
    p.add_argument("--raw-input", action="store_true", default=False)
    p.add_argument("--pose-mode", default="regression", choices=["classification", "regression"])
    p.add_argument("--rot-num-bins", type=int, default=72)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tactile-augment", action="store_true", default=False)
    p.add_argument("--gel-spin-deg", type=float, default=0.0)
    p.add_argument("--center-crop", action="store_true", default=False)
    p.add_argument("--use-obj-emb", action="store_true", default=True,
                   help="Add object embedding conditioning (matches VisTacFusion)")
    p.add_argument("--no-obj-emb", dest="use_obj_emb", action="store_false")
    p.add_argument("--num-objects", type=int, default=20,
                   help="Number of object classes for embedding table")
    p.add_argument("--kendall", action="store_true", default=False)
    p.add_argument("--w-rot",   type=float, default=1.0)
    p.add_argument("--w-trans", type=float, default=1.0)
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
    p.add_argument("--save-path", default="output_checkpoints/pose_dinov3")
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

    if args.raw_input:
        img_xform = T.Compose([T.ToTensor(), T.Normalize(mean=imagenet_mu, std=imagenet_std)])
    else:
        img_xform = T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)])

    # ── dataset ─────────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = PoseDataset(
        args.data_path, args.mesh_dir, img_xform,
        calibration_config=0,
        split="train", val_every=args.val_every,
        raw_input=args.raw_input,
        gel_spin_max_deg=args.gel_spin_deg,
        center_crop=args.center_crop,
        tactile_augment=args.tactile_augment)
    val_ds = PoseDataset(
        args.data_path, args.mesh_dir, img_xform,
        calibration_config=0,
        split="val", val_every=args.val_every,
        raw_input=args.raw_input,
        center_crop=args.center_crop,
        shared_obj_map=train_ds._obj_to_id)

    if args.real_path:
        from torch.utils.data import ConcatDataset
        real_train = PoseDataset(
            args.real_path, args.mesh_dir, img_xform,
            calibration_config=0, raw_input=True,
            split="train", val_every=args.val_every,
            center_crop=args.center_crop,
            shared_obj_map=train_ds._obj_to_id)
        real_val = PoseDataset(
            args.real_path, args.mesh_dir, img_xform,
            calibration_config=0, raw_input=True,
            split="val", val_every=args.val_every,
            center_crop=args.center_crop,
            shared_obj_map=train_ds._obj_to_id)
        oversample = args.real_oversample
        if oversample <= 0:
            oversample = max(1, len(train_ds) // max(1, len(real_train)) // 2)
        train_ds = ConcatDataset([train_ds] + [real_train] * oversample)
        val_ds = real_val
        print(f"  Co-training: sim + real×{oversample} = {len(train_ds)} train, "
              f"val={len(val_ds)} (real only)")

    from train_pose_sitr import CachedDataset
    val_ds = CachedDataset(val_ds, desc="Caching val", num_workers=args.num_workers)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=(args.num_workers > 0),
                              prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # ── encoder (frozen) ────────────────────────────────────────────────────
    print(f"\nBuilding DINOv3 encoder ({args.dinov3_model})...")
    encoder = DINOv3Encoder(
        model_name=args.dinov3_model, weights=args.dinov3_weights
    ).to(device)
    encoder.eval()
    embed_dim = encoder.embed_dim
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M "
          f"(frozen, embed_dim={embed_dim})")

    # ── pose head ───────────────────────────────────────────────────────────
    pose_head = PoseHead(
        dim=embed_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
        pose_mode=args.pose_mode, rot_num_bins=args.rot_num_bins,
    ).to(device)
    print(f"  Pose head: {sum(p.numel() for p in pose_head.parameters())/1e6:.1f}M "
          f"(mode={args.pose_mode}, bins={args.rot_num_bins})")

    # ── object embedding (optional, matches VisTacFusion) ────────────────
    obj_embedding = None
    if args.use_obj_emb:
        num_obj = max(args.num_objects, len(train_ds.objects))
        obj_embedding = nn.Embedding(num_obj, embed_dim).to(device)
        print(f"  Object embedding: {num_obj} classes, dim={embed_dim}")

    # ── optimizer ───────────────────────────────────────────────────────────
    pose_loss_fn = PoseLoss(args.pose_mode, args.rot_num_bins)

    log_vars = None
    if args.kendall:
        log_vars = nn.Parameter(torch.zeros(2, device=device))
        print("  Kendall uncertainty weighting: ON (rot + trans)")

    params = list(pose_head.parameters())
    if obj_embedding is not None:
        params.extend(obj_embedding.parameters())
    if log_vars is not None:
        params.append(log_vars)

    optimizer = torch.optim.AdamW(params, lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = WarmupThenPlateau(
        optimizer, args.warmup_epochs, warmup_lr=1e-6, base_lr=args.lr,
        factor=args.plateau_factor, plateau_patience=args.plateau_patience,
        min_lr=args.min_lr)
    scaler = GradScaler("cuda", enabled=args.amp)

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        pose_head.load_state_dict(ck["pose_head"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_val_loss = ck.get("best_val_loss", float("inf"))
        if "log_vars" in ck and log_vars is not None:
            log_vars.data = ck["log_vars"]
        if "obj_embedding" in ck and obj_embedding is not None:
            obj_embedding.load_state_dict(ck["obj_embedding"])
        print(f"  Resumed from epoch {start_epoch}")

    # ── training loop ───────────────────────────────────────────────────────
    log_path = osp.join(args.save_path, "train_log.csv")
    history = {"epochs": [], "train_loss": [], "l_rot": [],
               "l_trans": [], "val_loss": [], "lr": []}

    if start_epoch == 0:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,l_rot,l_trans,val_loss,v_rot,v_trans,rot_err_deg,trans_err,lr\n")

    epochs_no_improve = 0

    print("\n" + "=" * 60)
    print(f"Pose Training | encoder={args.dinov3_model}  mode={args.pose_mode}")
    input_mode = "RAW (ImageNet norm)" if args.raw_input else "DIFF (bg-sub)"
    print(f"  input={input_mode}  calib=0  gel_spin={args.gel_spin_deg}°  center_crop={args.center_crop}")
    print(f"  bs={args.batch_size}  lr={args.lr}  dropout={args.dropout}")
    obj_str = f"  obj_emb={args.use_obj_emb}" if args.use_obj_emb else ""
    loss_str = "Kendall (learned)" if args.kendall else f"w_rot={args.w_rot} w_trans={args.w_trans}"
    print(f"  loss: {loss_str}  early_stop={args.early_stop}  device={args.device}{obj_str}")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch:03d}/{args.epochs-1}  lr={lr:.2e}")

        train_loss, l_rot, l_trans = train_one_epoch(
            encoder, pose_head, train_loader, optimizer, scaler,
            pose_loss_fn, args.w_rot, args.w_trans,
            device, args.amp, epoch, log_vars=log_vars,
            obj_embedding=obj_embedding)

        val_loss, v_rot, v_trans, rot_err_deg, trans_err = validate(
            encoder, pose_head, val_loader, pose_loss_fn,
            args.w_rot, args.w_trans, device, args.amp, log_vars=log_vars,
            obj_embedding=obj_embedding)

        scheduler.step(val_loss)

        print(f"  → train={train_loss:.4f}  rot={l_rot:.4f}  trans={l_trans:.4f}")
        print(f"  → val={val_loss:.4f}  rot_err={rot_err_deg:.2f}°  trans_err={trans_err:.4f}")

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["l_rot"].append(l_rot)
        history["l_trans"].append(l_trans)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{l_rot:.6f},{l_trans:.6f},"
                    f"{val_loss:.6f},{v_rot:.6f},{v_trans:.6f},"
                    f"{rot_err_deg:.4f},{trans_err:.6f},{lr:.2e}\n")

        def save_ckpt(path):
            d = {
                "epoch": epoch,
                "pose_head": pose_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "args": vars(args),
            }
            if log_vars is not None:
                d["log_vars"] = log_vars.data
            if obj_embedding is not None:
                d["obj_embedding"] = obj_embedding.state_dict()
            torch.save(d, path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
