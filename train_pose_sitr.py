"""
train_pose_sitr.py — Train pose estimation head on frozen SITR encoder

SE(2) pose: (cos θ, sin θ, tx_norm, ty_norm)
- Rotation: classification into bins (cross-entropy + soft-argmax)
- Translation: L1 regression (normalized by object half-size)

Pose label convention matches VisTacFusion:
  delta_rz = session.base_rotation[2] - session_000.base_rotation[2]
  (sx, sy) rotated by delta_rz, normalized by half = target_size_mm / 2000

Usage:
  python train_pose_sitr.py \
    --data-path /media/hdd2/ihsuan/gs_blender/renders \
    --mesh-dir /media/hdd2/ihsuan/gs_blender/meshes \
    --encoder-weights output_checkpoints/20260706_sitr_finetune/best.pth \
    --device cuda:3
"""
import argparse
import glob as _glob
import json
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
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataloaders import sample_mu, sample_std, norm_mu, norm_std
from models.networks import SITR_base


# ── Pose Head (matches VisTacFusion) ────────────────────────────────────────

class PoseHead(nn.Module):
    def __init__(self, dim=768, hidden_dim=256, dropout=0.1,
                 pose_mode="classification", rot_num_bins=72,
                 use_spatial_pool=True):
        super().__init__()
        self.pose_mode = pose_mode
        self.rot_num_bins = rot_num_bins
        self.use_spatial_pool = use_spatial_pool

        in_dim = dim * 2 if use_spatial_pool else dim

        if pose_mode == "classification":
            self.rot_head = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, rot_num_bins),
            )
            self.trans_head = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
            bin_centers = torch.linspace(-math.pi, math.pi, rot_num_bins + 1)[:-1]
            bin_centers = bin_centers + (math.pi / rot_num_bins)
            self.register_buffer("bin_centers", bin_centers)
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 4),
            )

    def forward(self, cls_token, spatial_tokens=None):
        x = cls_token
        if self.use_spatial_pool and spatial_tokens is not None:
            pool = spatial_tokens.mean(dim=1)
            x = torch.cat([x, pool], dim=-1)

        if self.pose_mode == "classification":
            rot_logits = self.rot_head(x)
            trans = self.trans_head(x)
            probs = F.softmax(rot_logits, dim=-1)
            cos_v = (probs * torch.cos(self.bin_centers)).sum(dim=-1)
            sin_v = (probs * torch.sin(self.bin_centers)).sum(dim=-1)
            cos_sin = F.normalize(torch.stack([cos_v, sin_v], dim=-1), dim=-1)
            se2 = torch.cat([cos_sin, trans], dim=-1)
            return {"rot_logits": rot_logits, "se2": se2, "trans": trans}

        out = self.net(x)
        cos_sin = F.normalize(out[:, :2], dim=-1, eps=1e-6)
        return {"se2": torch.cat([cos_sin, out[:, 2:]], dim=-1)}


# ── Pose Loss (matches VisTacFusion) ────────────────────────────────────────

class PoseLoss(nn.Module):
    def __init__(self, pose_mode="classification", rot_num_bins=72):
        super().__init__()
        self.pose_mode = pose_mode
        self.rot_num_bins = rot_num_bins

    def _theta_to_bin(self, theta):
        bin_size = 2 * math.pi / self.rot_num_bins
        bins = ((theta + math.pi) / bin_size).long()
        return bins.clamp(0, self.rot_num_bins - 1)

    def forward(self, pred, gt_pose):
        cos_gt, sin_gt, txy_gt = gt_pose[:, 0], gt_pose[:, 1], gt_pose[:, 2:]
        if self.pose_mode == "classification":
            theta_gt = torch.atan2(sin_gt, cos_gt)
            target_bins = self._theta_to_bin(theta_gt)
            l_rot = F.cross_entropy(pred["rot_logits"], target_bins)
            l_trans = F.l1_loss(pred["trans"], txy_gt)
        else:
            se2 = pred["se2"]
            l_rot = (1.0 - (se2[:, 0] * cos_gt + se2[:, 1] * sin_gt)).mean()
            l_trans = F.l1_loss(se2[:, 2:], txy_gt)
        return l_rot, l_trans


# ── Dataset ─────────────────────────────────────────────────────────────────

class PoseDataset(Dataset):
    """Tactile images + SE(2) pose labels from gs_blender nested layout.
    Pose convention matches VisTacFusion: delta_rz from session_000, (x,y) rotated & normalized."""

    def __init__(self, root, mesh_dir, img_xform, calibration_config=18,
                 include_objects=None, split="all", val_every=20,
                 use_gt_norm=False, raw_input=False,
                 gel_spin_max_deg=0.0, center_crop=False,
                 tactile_augment=False):
        self.root = root
        self.img_xform = img_xform
        self.gel_spin_max_deg = gel_spin_max_deg if split == "train" else 0.0
        self.center_crop = center_crop
        self.tactile_augment_fn = None
        if tactile_augment and split == "train":
            from dataloaders import TactileAugment
            self.tactile_augment_fn = TactileAugment()
        self.calibration_config = calibration_config
        self.raw_input = raw_input
        self.norm_suffix = "_gt" if use_gt_norm else ""
        self._calib_cache = {}

        if calibration_config == 0:    self.calib_list = []
        elif calibration_config == 18: self.calib_list = list(range(1, 19))
        elif calibration_config == 19: self.calib_list = list(range(0, 19))
        else:                          self.calib_list = list(range(1, 19))

        obj_pose_info = self._load_object_pose_info(mesh_dir)

        units = sorted(_glob.glob(osp.join(root, "*", "session_*", "sensor_*")))
        units = [u for u in units if osp.isdir(osp.join(u, "samples"))]
        if include_objects is not None:
            incl = set(include_objects)
            units = [u for u in units
                     if osp.basename(osp.dirname(osp.dirname(u))) in incl]

        self.samples = []
        self.unit_meta = {}
        for unit in units:
            obj_name = osp.basename(osp.dirname(osp.dirname(unit)))
            info = obj_pose_info.get(obj_name)
            if info is None:
                continue

            session_dir = osp.dirname(unit)
            with open(osp.join(session_dir, "session.json")) as f:
                sess = json.load(f)
            delta_rz = sess["base_rotation"][2] - info["rz0"]

            self.unit_meta[unit] = {
                "delta_rz": delta_rz,
                "half": info["half"],
            }

            sample_dir = osp.join(unit, "samples")
            pngs = sorted(f for f in os.listdir(sample_dir) if f.endswith(".png"))
            for png in pngs:
                idx = int(osp.splitext(png)[0])
                pose_path = osp.join(unit, "raw_data", f"{idx:04d}_pose.json")
                if not osp.exists(pose_path):
                    continue
                is_val = (idx % val_every == 0)
                if split == "train" and is_val:
                    continue
                if split == "val" and not is_val:
                    continue
                self.samples.append((unit, idx))

        self.objects = sorted({osp.basename(osp.dirname(osp.dirname(u)))
                               for u in self.unit_meta})
        if not self.samples:
            raise RuntimeError(f"No samples found under {root}")
        print(f"  PoseDataset [{split}]: {len(self.samples)} samples, "
              f"{len(self.objects)} objects")

    def _load_object_pose_info(self, mesh_dir):
        import trimesh
        info = {}
        obj_dirs = sorted(d for d in os.listdir(self.root)
                          if osp.isdir(osp.join(self.root, d)))
        for obj_name in obj_dirs:
            mesh_path = osp.join(mesh_dir, f"{obj_name}.obj")
            s0_path = osp.join(self.root, obj_name, "session_000", "session.json")
            if not osp.exists(mesh_path) or not osp.exists(s0_path):
                continue
            mesh = trimesh.load(mesh_path, force="mesh")
            with open(s0_path) as f:
                d0 = json.load(f)
            target_size = d0.get("_target_size_mm", 82.0)
            info[obj_name] = {
                "half": target_size / 2.0 / 1000.0,
                "rz0": d0["base_rotation"][2],
            }
        return info

    def _get_calib(self, unit):
        cached = self._calib_cache.get(unit)
        if cached is None:
            cal_dir = osp.join(unit, "calibration")
            ref = np.array(Image.open(osp.join(cal_dir, "0000.png")))
            calib = [np.array(Image.open(osp.join(cal_dir, f"{i:04d}.png")))
                     for i in range(1, 19)]
            cached = (ref, calib)
            self._calib_cache[unit] = cached
        return cached

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        unit, sample_idx = self.samples[index]
        meta = self.unit_meta[unit]
        ref_img, calib_raw = self._get_calib(unit)

        sample = np.array(Image.open(
            osp.join(unit, "samples", f"{sample_idx:04d}.png")))

        if self.raw_input:
            sample_f = sample.astype(np.float32)
            all_imgs = [ref_img.astype(np.float32)] + [c.astype(np.float32) for c in calib_raw]
            calib_imgs = [all_imgs[i] for i in self.calib_list]
        else:
            ref_f = ref_img.astype(np.float32)
            sample_f = sample.astype(np.float32) - ref_f
            calib_imgs = [(calib_raw[i - 1].astype(np.float32) - ref_f)
                          for i in self.calib_list]

        # Gel-spin rotation (before center crop)
        rot_deg = 0.0
        if self.gel_spin_max_deg > 0:
            from dataloaders import gel_spin_rotate
            rot_deg = np.random.uniform(-self.gel_spin_max_deg, self.gel_spin_max_deg)
            sample_f, calib_imgs, _, _ = gel_spin_rotate(
                sample_f, calib_imgs, None, None, rot_deg)

        # Fixed center crop
        if self.center_crop:
            from dataloaders import fixed_center_crop
            sample_f = fixed_center_crop(sample_f)
            calib_imgs = [fixed_center_crop(c) for c in calib_imgs]

        # Photometric augmentation
        if self.tactile_augment_fn is not None:
            sample_f, calib_imgs, _, _ = self.tactile_augment_fn(
                sample_f, calib_imgs, None, None)

        sample_t = self.img_xform(sample_f)
        calib_t = torch.cat([self.img_xform(c) for c in calib_imgs]) if calib_imgs else torch.empty(0)

        # Pose label (VisTacFusion convention)
        with open(osp.join(unit, "raw_data", f"{sample_idx:04d}_pose.json")) as f:
            pdata = json.load(f)
        delta_rz = meta["delta_rz"]
        half = meta["half"]
        cos_rz, sin_rz = math.cos(delta_rz), math.sin(delta_rz)
        sx, sy = pdata["sample_x"], pdata["sample_y"]
        x_norm = (cos_rz * sx - sin_rz * sy) / max(half, 1e-8)
        y_norm = (sin_rz * sx + cos_rz * sy) / max(half, 1e-8)
        # Adjust pose theta for gel-spin rotation (θ' = θ - φ)
        if abs(rot_deg) > 0.01:
            phi_rad = math.radians(rot_deg)
            c, s = math.cos(-phi_rad), math.sin(-phi_rad)
            cos_new = cos_rz * c - sin_rz * s
            sin_new = sin_rz * c + cos_rz * s
            cos_rz, sin_rz = cos_new, sin_new
        pose = torch.tensor([cos_rz, sin_rz, x_norm, y_norm], dtype=torch.float32)

        return {"sample": sample_t, "calibration": calib_t, "pose": pose}


# ── Helpers ─────────────────────────────────────────────────────────────────

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


def plot_loss_curves(history, save_path):
    epochs = history["epochs"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Pose Head Training Curves", fontsize=14, fontweight="bold")

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


# ── Train / Val ─────────────────────────────────────────────────────────────

def train_one_epoch(encoder, pose_head, loader, optimizer, scaler,
                    pose_loss_fn, w_rot, w_trans, device, amp_enabled, epoch,
                    log_vars=None):
    pose_head.train()
    loss_m, rot_m, trans_m = AverageMeter(), AverageMeter(), AverageMeter()
    t0 = time.time()

    for step, batch in enumerate(loader):
        imgs   = batch["sample"].to(device, non_blocking=True)
        calibs = batch["calibration"].to(device, non_blocking=True)
        gt_pose = batch["pose"].to(device, non_blocking=True)
        B = imgs.size(0)

        optimizer.zero_grad()
        with autocast("cuda", enabled=amp_enabled):
            with torch.no_grad():
                enc_out = encoder(imgs, calibs)
            cls_token = enc_out["latent"][:, 0, :]
            spatial = enc_out["latent"][:, 1:, :]
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
             device, amp_enabled, log_vars=None):
    pose_head.eval()
    loss_m, rot_m, trans_m = AverageMeter(), AverageMeter(), AverageMeter()
    all_se2_pred, all_se2_gt = [], []

    for batch in loader:
        imgs   = batch["sample"].to(device, non_blocking=True)
        calibs = batch["calibration"].to(device, non_blocking=True)
        gt_pose = batch["pose"].to(device, non_blocking=True)

        with autocast("cuda", enabled=amp_enabled):
            enc_out = encoder(imgs, calibs)
            cls_token = enc_out["latent"][:, 0, :]
            spatial = enc_out["latent"][:, 1:, :]
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

    # Compute rotation error in degrees
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
    p.add_argument("--encoder-weights", required=True)
    p.add_argument("--calibration-config", type=int, default=18)
    p.add_argument("--val-every", type=int, default=20)
    p.add_argument("--pose-mode", default="regression", choices=["classification", "regression"])
    p.add_argument("--rot-num-bins", type=int, default=72)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tactile-augment", action="store_true", default=False)
    p.add_argument("--gel-spin-deg", type=float, default=0.0)
    p.add_argument("--center-crop", action="store_true", default=False)
    p.add_argument("--kendall", action="store_true", default=False)
    p.add_argument("--w-rot",   type=float, default=1.0)
    p.add_argument("--w-trans", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
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
    p.add_argument("--save-path", default="output_checkpoints/20260707_pose_sitr")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)

    import torchvision.transforms as T
    from dataloaders import imagenet_mu, imagenet_std
    if args.calibration_config == 0:
        img_xform = T.Compose([T.ToTensor(), T.Normalize(mean=imagenet_mu, std=imagenet_std)])
    else:
        img_xform = T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)])

    # ── dataset ─────────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = PoseDataset(
        args.data_path, args.mesh_dir, img_xform,
        calibration_config=args.calibration_config,
        split="train", val_every=args.val_every,
        gel_spin_max_deg=args.gel_spin_deg,
        center_crop=args.center_crop,
        tactile_augment=args.tactile_augment)
    val_ds = PoseDataset(
        args.data_path, args.mesh_dir, img_xform,
        calibration_config=args.calibration_config,
        split="val", val_every=args.val_every,
        center_crop=args.center_crop)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=(args.num_workers > 0),
                              prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=(args.num_workers > 0),
                            prefetch_factor=4 if args.num_workers > 0 else None)

    # ── encoder (frozen) ────────────────────────────────────────────────────
    print(f"Loading SITR encoder from {args.encoder_weights}...")
    encoder = SITR_base(num_calibration=args.calibration_config).to(device)
    state = torch.load(args.encoder_weights, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    encoder.load_state_dict(state, strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M (frozen)")

    # ── pose head ───────────────────────────────────────────────────────────
    pose_head = PoseHead(
        dim=768, hidden_dim=args.hidden_dim, dropout=args.dropout,
        pose_mode=args.pose_mode, rot_num_bins=args.rot_num_bins,
    ).to(device)
    print(f"  Pose head: {sum(p.numel() for p in pose_head.parameters())/1e6:.1f}M "
          f"(mode={args.pose_mode}, bins={args.rot_num_bins})")

    # ── optimizer ───────────────────────────────────────────────────────────
    pose_loss_fn = PoseLoss(args.pose_mode, args.rot_num_bins)

    log_vars = None
    if args.kendall:
        log_vars = nn.Parameter(torch.zeros(2, device=device))
        print("  Kendall uncertainty weighting: ON (rot + trans)")

    params = list(pose_head.parameters())
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
    print(f"Pose Training | encoder=SITR  mode={args.pose_mode}  bins={args.rot_num_bins}")
    print(f"  bs={args.batch_size}  lr={args.lr}  dropout={args.dropout}")
    loss_str = "Kendall (learned)" if args.kendall else f"w_rot={args.w_rot} w_trans={args.w_trans}"
    print(f"  loss: {loss_str}  early_stop={args.early_stop}  device={args.device}")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch:03d}/{args.epochs-1}  lr={lr:.2e}")

        train_loss, l_rot, l_trans = train_one_epoch(
            encoder, pose_head, train_loader, optimizer, scaler,
            pose_loss_fn, args.w_rot, args.w_trans,
            device, args.amp, epoch, log_vars=log_vars)

        val_loss, v_rot, v_trans, rot_err_deg, trans_err = validate(
            encoder, pose_head, val_loader, pose_loss_fn,
            args.w_rot, args.w_trans, device, args.amp, log_vars=log_vars)

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
