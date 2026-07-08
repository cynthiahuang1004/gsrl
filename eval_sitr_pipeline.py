"""
eval_sitr_pipeline.py — Evaluate SITR encoder + DPT decoder + Pose head on sim val set

Computes quantitative metrics matching VisTacFusion:
  - Depth: MSE, MAE, RMSE, delta<1.25
  - Normal: MSE, angular error (mean/median)
  - Pose: rotation error (deg), translation error

Usage:
  python eval_sitr_pipeline.py \
    --data-path /media/hdd2/ihsuan/gs_blender/renders \
    --mesh-dir /media/hdd2/ihsuan/gs_blender/meshes \
    --encoder-weights output_checkpoints/20260706_sitr_finetune/best.pth \
    --dpt-weights output_checkpoints/20260707_sitr_dpt/best.pth \
    --pose-weights output_checkpoints/20260707_pose_sitr/best.pth \
    --save-path eval_results/sitr_pipeline \
    --device cuda:2
"""
import argparse
import json
import math
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.networks import SITR_base
from models.dpt import SITRWithDPT
from train_pose_sitr import PoseHead, PoseDataset
from dataloaders import (sim_dataset_nested, sample_mu, sample_std,
                         norm_mu, norm_std, dmap_mu, dmap_std)
import torchvision.transforms as T


# ── Metrics (matching VisTacFusion) ─────────────────────────────────────────

def _angular_error_deg(pred_normal, gt_normal):
    """Both: (H, W, 3) unit vectors [-1,1]. Returns (mean_deg, median_deg)."""
    p = pred_normal / (np.linalg.norm(pred_normal, axis=-1, keepdims=True) + 1e-8)
    g = gt_normal / (np.linalg.norm(gt_normal, axis=-1, keepdims=True) + 1e-8)
    cos_sim = np.clip((p * g).sum(axis=-1), -1, 1)
    angles = np.degrees(np.arccos(cos_sim))
    return float(np.mean(angles)), float(np.median(angles))


def _depth_metrics(pred, gt):
    """Both: (H, W). MSE = full-image; MAE/RMSE/delta = contact-only (gt > 0).
    Matches VisTacFusion convention."""
    mse_full = float(((pred - gt) ** 2).mean())
    mask = gt > 0
    contact = {}
    if mask.sum() > 0:
        p, g = pred[mask], gt[mask]
        abs_diff = np.abs(p - g)
        contact["MAE"] = float(abs_diff.mean())
        contact["RMSE"] = float((abs_diff ** 2).mean()) ** 0.5
        ratio = np.maximum(p / np.clip(g, 1e-6, None), g / np.clip(p, 1e-6, None))
        contact["delta<1.25"] = float((ratio < 1.25).mean()) * 100
    return {"MSE": mse_full, **contact}


def _rotation_error_deg(pred_pose, gt_pose):
    """pred/gt: [cos, sin, tx, ty]. Returns error in degrees."""
    theta_pred = np.arctan2(pred_pose[1], pred_pose[0])
    theta_gt = np.arctan2(gt_pose[1], gt_pose[0])
    err = abs(theta_pred - theta_gt)
    err = min(err, 2 * np.pi - err)
    return float(np.degrees(err))


def unnorm(tensor, mu, std):
    mu = torch.tensor(mu, dtype=tensor.dtype).view(1, -1, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype).view(1, -1, 1, 1)
    return tensor * std + mu


# ── Real data inference ─────────────────────────────────────────────────────

IMG_SIZE = 224
CALIB_LIST = list(range(1, 19))


def load_resized(path, size=IMG_SIZE):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def build_real_calib(calib_from, tf):
    cal_dir = osp.join(calib_from, "calibration")
    ref = load_resized(osp.join(cal_dir, "0000.png"))
    chans = []
    for i in CALIB_LIST:
        c = load_resized(osp.join(cal_dir, f"{i:04d}.png"))
        chans.append(tf(c - ref))
    return torch.cat(chans, dim=0).unsqueeze(0)


@torch.no_grad()
def eval_real(model, encoder, pose_head, real_dir, calib_from,
              num_samples, device, save_dir):
    """Run inference on real tactile images (qualitative — no GT)."""
    os.makedirs(save_dir, exist_ok=True)

    tf = T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)])

    tac_dir = osp.join(real_dir, "tactile_images")
    base_dir = osp.join(real_dir, "base_tactile_images")

    files = sorted([f for f in os.listdir(tac_dir) if f.endswith(".jpg") or f.endswith(".png")])
    step = max(1, len(files) // num_samples)
    files = files[::step][:num_samples]
    print(f"\nReal data: {len(files)} samples from {real_dir}")

    calib = build_real_calib(calib_from, tf).to(device)

    for i, fname in enumerate(tqdm(files, desc="Eval Real")):
        name = osp.splitext(fname)[0]
        tac_img = load_resized(osp.join(tac_dir, fname))

        base_path = osp.join(base_dir, fname)
        if osp.exists(base_path):
            base_img = load_resized(base_path)
        else:
            base_img = load_resized(osp.join(base_dir, os.listdir(base_dir)[0]))

        diff = tac_img - base_img
        sample_t = tf(diff).unsqueeze(0).to(device)
        calib_t = calib.expand(1, -1, -1, -1)

        with autocast("cuda"):
            out = model(sample_t, calib_t)

        pred_depth = out["depth"][0, 0].cpu().numpy()
        pred_norm = unnorm(out["normal"].cpu(), norm_mu, norm_std)
        pred_norm = pred_norm[0].permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)

        # Pose
        pose_str = ""
        if pose_head is not None and encoder is not None:
            with autocast("cuda"):
                latent = encoder.forward_encoder(sample_t, calib_t)
                cls_tok = latent[:, 0, :]
                spatial = latent[:, 1:, :]
                pred_pose = pose_head(cls_tok, spatial)
            se2 = pred_pose["se2"][0].float().cpu().numpy()
            theta = np.degrees(np.arctan2(se2[1], se2[0]))
            pose_str = f"  θ={theta:.1f}°  tx={se2[2]:.3f}  ty={se2[3]:.3f}"

        # Visualize
        raw_img = tac_img.astype(np.uint8)
        diff_vis = np.clip((diff - diff.min()) / (np.ptp(diff) + 1e-6) * 255, 0, 255).astype(np.uint8)

        ncol = 4 + (1 if pose_str else 0)
        fig, axes = plt.subplots(1, ncol, figsize=(4 * ncol, 4))
        axes[0].imshow(raw_img); axes[0].set_title(f"Real ({name})")
        axes[1].imshow(diff_vis); axes[1].set_title("Bg-subtracted")
        axes[2].imshow(pred_norm); axes[2].set_title("Pred Normal")
        im = axes[3].imshow(pred_depth, cmap="viridis")
        axes[3].set_title("Pred Depth")
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        if pose_str:
            axes[4].text(0.5, 0.5, pose_str, transform=axes[4].transAxes,
                         fontsize=16, ha="center", va="center", family="monospace")
            axes[4].set_title("Pose")
            axes[4].axis("off")
        for ax in axes[:4]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(osp.join(save_dir, f"real_{i:03d}_{name}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"  Real visualizations saved -> {save_dir}/")


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="/media/hdd2/ihsuan/gs_blender/renders")
    p.add_argument("--mesh-dir",  default="/media/hdd2/ihsuan/gs_blender/meshes")
    p.add_argument("--encoder-weights", required=True)
    p.add_argument("--dpt-weights", required=True)
    p.add_argument("--pose-weights", default=None)
    p.add_argument("--calibration-config", type=int, default=18)
    p.add_argument("--val-every", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-vis", type=int, default=30)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save-path", default="eval_results/sitr_pipeline")
    p.add_argument("--real-dir", default=None,
                   help="Real data dir with tactile_images/ and base_tactile_images/")
    p.add_argument("--calib-from", default=None,
                   help="Sim sensor dir to borrow calibration from (for real data)")
    p.add_argument("--num-real", type=int, default=30)
    p.add_argument("--center-crop", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)

    img_xform  = T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)])
    dmap_xform = T.Compose([T.ToTensor(), T.Normalize(mean=dmap_mu, std=dmap_std)])
    norm_xform = T.Compose([T.ToTensor(), T.Normalize(mean=norm_mu, std=norm_std)])

    # ── DPT eval dataset ────────────────────────────────────────────────────
    print("Loading DPT val dataset...")
    full_ds = sim_dataset_nested(
        path=args.data_path, augment=False,
        transforms=img_xform, dmap_transforms=dmap_xform, norm_transforms=norm_xform,
        calibration_config=args.calibration_config, sendTwo=False,
        use_gt_norm=True, raw_input=False,
        center_crop=args.center_crop,
    )
    spu = full_ds.samples_per_unit
    val_idx = [i for i in range(len(full_ds)) if (i % spu) % args.val_every == 0]
    val_ds = torch.utils.data.Subset(full_ds, val_idx)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"  Val: {len(val_ds)} samples")

    # ── Load encoder + DPT ──────────────────────────────────────────────────
    print(f"Loading SITR encoder from {args.encoder_weights}...")
    sitr = SITR_base(num_calibration=args.calibration_config)
    enc_state = torch.load(args.encoder_weights, map_location="cpu", weights_only=False)
    if isinstance(enc_state, dict) and "model" in enc_state:
        enc_state = enc_state["model"]
    sitr.load_state_dict(enc_state, strict=False)

    model = SITRWithDPT(sitr, embed_dim=768, features=256).to(device)

    print(f"Loading DPT decoder from {args.dpt_weights}...")
    dpt_ck = torch.load(args.dpt_weights, map_location="cpu", weights_only=False)
    model.decoder.load_state_dict(dpt_ck["decoder"])
    model.eval()

    # ── Load pose head ──────────────────────────────────────────────────────
    pose_head = None
    if args.pose_weights:
        print(f"Loading pose head from {args.pose_weights}...")
        pose_ck = torch.load(args.pose_weights, map_location="cpu", weights_only=False)
        pose_args = pose_ck.get("args", {})
        pose_head = PoseHead(
            dim=768,
            hidden_dim=pose_args.get("hidden_dim", 256),
            dropout=0.0,
            pose_mode=pose_args.get("pose_mode", "classification"),
            rot_num_bins=pose_args.get("rot_num_bins", 72),
        ).to(device)
        pose_head.load_state_dict(pose_ck["pose_head"])
        pose_head.eval()

    # ── Pick vis indices: one per object, evenly spaced within each ────────
    spu = full_ds.samples_per_unit
    obj_vis_indices = {}
    for vi in val_idx:
        unit_idx = vi // spu
        unit = full_ds.units[unit_idx]
        obj = osp.basename(osp.dirname(osp.dirname(unit)))
        if obj not in obj_vis_indices:
            obj_vis_indices[obj] = vi
    vis_global_set = set(obj_vis_indices.values())
    print(f"  Will visualize {len(vis_global_set)} samples (1 per object)")

    # ── Full quantitative eval (per-sample, matching VisTacFusion) ──────────
    print("\nEvaluating (depth + normal + pose, per-sample)...")

    metrics = {
        "depth_mse": [], "depth_mae": [], "depth_rmse": [], "depth_d1": [],
        "normal_mse": [], "normal_ang_mean": [], "normal_ang_median": [],
        "pose_rot_deg": [], "pose_trans_l1": [],
    }

    encoder_raw = model.encoder.sitr
    encoder_raw.eval()

    # Pose dataset (for GT pose labels)
    pose_ds = None
    if pose_head is not None:
        pose_ds = PoseDataset(
            args.data_path, args.mesh_dir, img_xform,
            calibration_config=args.calibration_config,
            split="val", val_every=args.val_every)

    sample_counter = 0
    vis_dir = osp.join(args.save_path, "sim_val_vis")
    os.makedirs(vis_dir, exist_ok=True)

    for batch in tqdm(val_loader, desc="Eval"):
        imgs   = batch["sample"].to(device)
        calibs = batch["calibration"].to(device)
        gt_depth_t = batch["dmap"]
        gt_norm_t  = batch["norm"]
        B = imgs.size(0)

        with torch.no_grad(), autocast("cuda"):
            out = model(imgs, calibs)

        pred_depth_t = out["depth"].cpu()
        pred_norm_t  = out["normal"].cpu()

        # Unnormalize to pixel space
        pred_d_un = unnorm(pred_depth_t, dmap_mu, dmap_std)
        gt_d_un   = unnorm(gt_depth_t, dmap_mu, dmap_std)
        pred_n_un = unnorm(pred_norm_t, norm_mu, norm_std)
        gt_n_un   = unnorm(gt_norm_t, norm_mu, norm_std)

        for i in range(B):
            pd = pred_d_un[i, 0].numpy()
            gd = gt_d_un[i, 0].numpy()
            dm = _depth_metrics(pd, gd)
            if dm:
                metrics["depth_mse"].append(dm["MSE"])
                metrics["depth_mae"].append(dm["MAE"])
                metrics["depth_rmse"].append(dm["RMSE"])
                metrics["depth_d1"].append(dm["delta<1.25"])

            # Normal metrics (unit vectors)
            pn = pred_n_un[i].permute(1, 2, 0).numpy() / 127.5 - 1.0
            gn = gt_n_un[i].permute(1, 2, 0).numpy() / 127.5 - 1.0
            nm = F.mse_loss(torch.from_numpy(pn).float(),
                            torch.from_numpy(gn).float()).item()
            metrics["normal_mse"].append(nm)
            ang_mean, ang_median = _angular_error_deg(pn, gn)
            metrics["normal_ang_mean"].append(ang_mean)
            metrics["normal_ang_median"].append(ang_median)

            # Pose metrics
            if pose_head is not None and pose_ds is not None and sample_counter + i < len(pose_ds):
                pose_sample = pose_ds[sample_counter + i]
                gt_pose = pose_sample["pose"].numpy()
                p_imgs = pose_sample["sample"].unsqueeze(0).to(device)
                p_cals = pose_sample["calibration"].unsqueeze(0).to(device)
                with torch.no_grad(), autocast("cuda"):
                    latent = encoder_raw.forward_encoder(p_imgs, p_cals)
                    pred_pose = pose_head(latent[:, 0, :], latent[:, 1:, :])
                se2 = pred_pose["se2"][0].float().cpu().numpy()
                metrics["pose_rot_deg"].append(_rotation_error_deg(se2, gt_pose))
                metrics["pose_trans_l1"].append(float(np.abs(se2[2:] - gt_pose[2:]).mean()))

            # Visualization (one per object)
            global_idx = val_idx[sample_counter + i]
            if global_idx in vis_global_set:
                unit_idx = global_idx // spu
                unit = full_ds.units[unit_idx]
                obj_name = osp.basename(osp.dirname(osp.dirname(unit)))

                fig, axes = plt.subplots(1, 6, figsize=(24, 4))
                inp = unnorm(imgs[i:i+1].cpu(), sample_mu, sample_std)
                inp = inp[0].permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)
                axes[0].imshow(inp); axes[0].set_title(f"Input [{obj_name}]")

                vmin = min(pd.min(), gd.min())
                vmax = max(pd.max(), gd.max())
                axes[1].imshow(pd, cmap="viridis", vmin=vmin, vmax=vmax); axes[1].set_title("Pred Depth")
                axes[2].imshow(gd, cmap="viridis", vmin=vmin, vmax=vmax); axes[2].set_title("GT Depth")

                pn_vis = ((pn * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                gn_vis = ((gn * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                axes[3].imshow(pn_vis); axes[3].set_title("Pred Normal")
                axes[4].imshow(gn_vis); axes[4].set_title("GT Normal")

                err = np.abs(pd - gd)
                axes[5].imshow(err, cmap="hot"); axes[5].set_title(f"Depth Err (MAE={err.mean():.2f})")

                for ax in axes:
                    ax.axis("off")
                fig.tight_layout()
                fig.savefig(osp.join(vis_dir, f"{obj_name}.png"), dpi=120, bbox_inches="tight")
                plt.close(fig)

        sample_counter += B

    # ── Aggregate & print (matching VisTacFusion format) ────────────────────
    summary = {k: float(np.mean(v)) for k, v in metrics.items() if v}

    print("\n" + "=" * 60)
    print("SITR Pipeline Evaluation Results")
    print("=" * 60)

    report_path = osp.join(args.save_path, "eval_metrics.txt")
    with open(report_path, "w") as f:
        f.write("SITR Pipeline Evaluation Results\n")
        f.write(f"Encoder: {args.encoder_weights}\n")
        f.write(f"DPT: {args.dpt_weights}\n")
        if args.pose_weights:
            f.write(f"Pose: {args.pose_weights}\n")
        f.write(f"Val samples: {len(val_ds)}\n\n")

        f.write("[tactile]\n")
        f.write(f"  Depth MSE:               {summary.get('depth_mse', float('nan')):.6f}\n")
        f.write(f"  Depth MAE:               {summary.get('depth_mae', float('nan')):.6f}\n")
        f.write(f"  Depth RMSE:              {summary.get('depth_rmse', float('nan')):.6f}\n")
        f.write(f"  Depth delta<1.25 (%):    {summary.get('depth_d1', float('nan')):.2f}\n")
        f.write(f"  Normal MSE:              {summary.get('normal_mse', float('nan')):.6f}\n")
        f.write(f"  Normal Ang Mean (deg):   {summary.get('normal_ang_mean', float('nan')):.2f}\n")
        f.write(f"  Normal Ang Median (deg): {summary.get('normal_ang_median', float('nan')):.2f}\n")
        f.write(f"  Pose Rot Error (deg):    {summary.get('pose_rot_deg', float('nan')):.2f}\n")
        f.write(f"  Pose Trans L1:           {summary.get('pose_trans_l1', float('nan')):.6f}\n")

    # Also print to console
    with open(report_path) as f:
        print(f.read())

    print(f"Metrics saved -> {report_path}")
    print(f"Visualizations saved -> {vis_dir}/")

    # ── Real data eval (qualitative) ────────────────────────────────────────
    if args.real_dir:
        calib_from = args.calib_from or osp.join(
            args.data_path, "edge", "session_000", "sensor_0000")
        encoder_raw = model.encoder.sitr
        real_save = osp.join(args.save_path, "real")
        eval_real(model, encoder_raw, pose_head, args.real_dir, calib_from,
                  args.num_real, device, real_save)


if __name__ == "__main__":
    main()
