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
              num_samples, device, save_dir, raw_input=False,
              calibration_config=18, depth_from_npy=False, center_crop=False):
    """Run inference on real tactile images (qualitative — no GT)."""
    from dataloaders import imagenet_mu, imagenet_std, fixed_center_crop
    os.makedirs(save_dir, exist_ok=True)

    if raw_input:
        tf = T.Compose([T.ToTensor(), T.Normalize(mean=imagenet_mu, std=imagenet_std)])
    else:
        tf = T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)])

    tac_dir = osp.join(real_dir, "tactile_images")
    base_dir = osp.join(real_dir, "base_tactile_images")

    files = sorted([f for f in os.listdir(tac_dir) if f.endswith(".jpg") or f.endswith(".png")])
    step = max(1, len(files) // num_samples)
    files = files[::step][:num_samples]
    print(f"\nReal data: {len(files)} samples from {real_dir}")

    if calibration_config > 0:
        calib = build_real_calib(calib_from, tf).to(device)
    else:
        calib = torch.zeros(1, 0, IMG_SIZE, IMG_SIZE, device=device)

    for i, fname in enumerate(tqdm(files, desc="Eval Real")):
        name = osp.splitext(fname)[0]
        tac_img = load_resized(osp.join(tac_dir, fname))

        base_path = osp.join(base_dir, fname)
        if osp.exists(base_path):
            base_img = load_resized(base_path)
        else:
            base_img = load_resized(osp.join(base_dir, os.listdir(base_dir)[0]))

        if raw_input:
            inp = tac_img
        else:
            inp = tac_img - base_img

        if center_crop:
            inp = fixed_center_crop(inp, out_size=IMG_SIZE)

        sample_t = tf(inp).unsqueeze(0).to(device)
        calib_t = calib.expand(1, -1, -1, -1)

        with autocast("cuda"):
            out = model(sample_t, calib_t)

        pred_depth = out["depth"][0, 0].cpu().numpy()
        if depth_from_npy:
            pred_norm_vis = ((out["normal"][0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        else:
            pred_norm = unnorm(out["normal"].cpu(), norm_mu, norm_std)
            pred_norm_vis = pred_norm[0].permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)

        # Pose
        pose_str = ""
        if pose_head is not None and encoder is not None:
            with autocast("cuda"):
                if hasattr(encoder, 'forward_encoder'):
                    latent = encoder.forward_encoder(sample_t, calib_t)
                else:
                    latent = encoder(sample_t, calib_t)["latent"]
                cls_tok = latent[:, 0, :]
                spatial = latent[:, 1:, :]
                pred_pose = pose_head(cls_tok, spatial)
            se2 = pred_pose["se2"][0].float().cpu().numpy()
            theta = np.degrees(np.arctan2(se2[1], se2[0]))
            pose_str = f"  θ={theta:.1f}°  tx={se2[2]:.3f}  ty={se2[3]:.3f}"

        # Visualize
        raw_img = tac_img.astype(np.uint8)

        ncol = 4 + (1 if pose_str else 0)
        fig, axes = plt.subplots(1, ncol, figsize=(4 * ncol, 4))
        axes[0].imshow(raw_img); axes[0].set_title(f"Real ({name})")
        axes[1].imshow(inp.astype(np.uint8) if raw_input else
                       np.clip((inp - inp.min()) / (np.ptp(inp) + 1e-6) * 255, 0, 255).astype(np.uint8))
        axes[1].set_title("Input" if raw_input else "Bg-subtracted")
        axes[2].imshow(pred_norm_vis); axes[2].set_title("Pred Normal")
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
    p.add_argument("--save-path", default=None,
                   help="Output dir. Default: eval_results/<dpt-weights folder name>")
    p.add_argument("--real-dir", default=None,
                   help="Real data dir with tactile_images/ and base_tactile_images/")
    p.add_argument("--calib-from", default=None,
                   help="Sim sensor dir to borrow calibration from (for real data)")
    p.add_argument("--num-real", type=int, default=30)
    p.add_argument("--center-crop", action="store_true", default=False)
    p.add_argument("--raw-input", action="store_true", default=False)
    p.add_argument("--depth-from-npy", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.save_path is None:
        ckpt_dir_name = osp.basename(osp.dirname(args.dpt_weights))
        args.save_path = osp.join("eval_results", ckpt_dir_name)
    os.makedirs(args.save_path, exist_ok=True)

    from dataloaders import imagenet_mu, imagenet_std
    if args.raw_input:
        img_xform = T.Compose([T.ToTensor(), T.Normalize(mean=imagenet_mu, std=imagenet_std)])
    else:
        img_xform = T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)])
    dmap_xform = T.Compose([T.ToTensor(), T.Normalize(mean=dmap_mu, std=dmap_std)])
    norm_xform = T.Compose([T.ToTensor(), T.Normalize(mean=norm_mu, std=norm_std)])

    # ── DPT eval dataset ────────────────────────────────────────────────────
    print("Loading DPT val dataset...")
    full_ds = sim_dataset_nested(
        path=args.data_path, augment=False,
        transforms=img_xform, dmap_transforms=dmap_xform, norm_transforms=norm_xform,
        calibration_config=args.calibration_config, sendTwo=False,
        use_gt_norm=True, raw_input=args.raw_input,
        center_crop=args.center_crop,
        depth_from_npy=args.depth_from_npy,
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
    obj_embedding = None
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

        if "obj_embedding" in pose_ck:
            num_obj, emb_dim = pose_ck["obj_embedding"]["weight"].shape
            obj_embedding = nn.Embedding(num_obj, emb_dim)
            obj_embedding.load_state_dict(pose_ck["obj_embedding"])
            obj_embedding = obj_embedding.to(device).eval()
            print(f"  Object embedding loaded: {obj_embedding.num_embeddings} classes")

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

    # ── Load pose GT labels (batched) ────────────────────────────────────────
    pose_gt_map = {}
    pose_obj_map = {}
    if pose_head is not None:
        pose_ds = PoseDataset(
            args.data_path, args.mesh_dir, img_xform,
            calibration_config=args.calibration_config,
            split="val", val_every=args.val_every)
        for pi in range(len(pose_ds)):
            sample = pose_ds[pi]
            pose_gt_map[pi] = sample["pose"].numpy()
            if obj_embedding is not None:
                pose_obj_map[pi] = sample["object"]

    # ── Full quantitative eval (batched, encoder runs once) ────────────────
    print("\nEvaluating (depth + normal + pose, batched)...")

    metrics = {
        "depth_mse": [], "depth_mae": [], "depth_rmse": [], "depth_d1": [],
        "normal_mse": [], "normal_ang_mean": [], "normal_ang_median": [],
        "pose_rot_deg": [], "pose_rot_loss": [], "pose_rot_l1": [],
        "pose_trans_l1": [],
    }

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
            if pose_head is not None:
                features, latent = model.forward_encoder_full(imgs, calibs)
                depth, normal = model.decoder(features)
                out = {"depth": depth, "normal": normal}
                if obj_embedding is not None:
                    obj_ids = torch.tensor(
                        [pose_obj_map[sample_counter + j] for j in range(B)],
                        device=device)
                    latent = latent + obj_embedding(obj_ids).unsqueeze(1)
                pred_pose = pose_head(latent[:, 0, :], latent[:, 1:, :])
                pred_se2 = pred_pose["se2"].float().cpu().numpy()
            else:
                out = model(imgs, calibs)
                pred_se2 = None

        pred_depth_t = out["depth"].cpu()
        pred_norm_t  = out["normal"].cpu()

        if args.depth_from_npy:
            pred_d = pred_depth_t
            gt_d = gt_depth_t
            pred_n = pred_norm_t
            gt_n = gt_norm_t
        else:
            pred_d = unnorm(pred_depth_t, dmap_mu, dmap_std)
            gt_d = unnorm(gt_depth_t, dmap_mu, dmap_std)
            pred_n = unnorm(pred_norm_t, norm_mu, norm_std) / 127.5 - 1.0
            gt_n = unnorm(gt_norm_t, norm_mu, norm_std) / 127.5 - 1.0

        for i in range(B):
            pd = pred_d[i, 0].numpy()
            gd = gt_d[i, 0].numpy()
            dm = _depth_metrics(pd, gd)
            metrics["depth_mse"].append(dm["MSE"])
            if "MAE" in dm:
                metrics["depth_mae"].append(dm["MAE"])
                metrics["depth_rmse"].append(dm["RMSE"])
                metrics["depth_d1"].append(dm["delta<1.25"])

            pn = pred_n[i].permute(1, 2, 0).numpy()
            gn = gt_n[i].permute(1, 2, 0).numpy()
            nm = F.mse_loss(torch.from_numpy(pn).float(),
                            torch.from_numpy(gn).float()).item()
            metrics["normal_mse"].append(nm)
            ang_mean, ang_median = _angular_error_deg(pn, gn)
            metrics["normal_ang_mean"].append(ang_mean)
            metrics["normal_ang_median"].append(ang_median)

            if pred_se2 is not None and sample_counter + i in pose_gt_map:
                gt_pose = pose_gt_map[sample_counter + i]
                se2 = pred_se2[i]
                metrics["pose_rot_deg"].append(_rotation_error_deg(se2, gt_pose))
                rot_loss = 1.0 - (se2[0] * gt_pose[0] + se2[1] * gt_pose[1])
                metrics["pose_rot_loss"].append(float(rot_loss))
                rot_l1 = abs(se2[0] - gt_pose[0]) + abs(se2[1] - gt_pose[1])
                metrics["pose_rot_l1"].append(float(rot_l1))
                metrics["pose_trans_l1"].append(float(np.abs(se2[2:] - gt_pose[2:]).mean()))

            global_idx = val_idx[sample_counter + i]
            if global_idx in vis_global_set:
                unit_idx = global_idx // spu
                unit = full_ds.units[unit_idx]
                obj_name = osp.basename(osp.dirname(osp.dirname(unit)))

                fig, axes = plt.subplots(1, 6, figsize=(24, 4))
                inp_mu = imagenet_mu if args.raw_input else sample_mu
                inp_std = imagenet_std if args.raw_input else sample_std
                inp = unnorm(imgs[i:i+1].cpu(), inp_mu, inp_std)
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
        f.write(f"  Pose Rot Loss (1-cos):   {summary.get('pose_rot_loss', float('nan')):.6f}\n")
        f.write(f"  Pose Rot L1 (cos/sin):   {summary.get('pose_rot_l1', float('nan')):.6f}\n")
        f.write(f"  Pose Trans L1:           {summary.get('pose_trans_l1', float('nan')):.6f}\n")

    with open(report_path) as f:
        print(f.read())

    print(f"Metrics saved -> {report_path}")
    print(f"Visualizations saved -> {vis_dir}/")

    # ── Pose error histogram ──────────────────────────────────────────────
    if metrics["pose_rot_deg"]:
        rot_deg = np.array(metrics["pose_rot_deg"])
        trans_err = np.array(metrics["pose_trans_l1"])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.hist(rot_deg, bins=60, range=(0, 180), color="#4C72B0",
                edgecolor="white", alpha=0.85)
        ax.axvline(rot_deg.mean(), color="red", ls="--",
                   label=f"Mean: {rot_deg.mean():.1f}°")
        ax.axvline(np.median(rot_deg), color="orange", ls="--",
                   label=f"Median: {np.median(rot_deg):.1f}°")
        ax.set_xlabel("Rotation Error (degrees)")
        ax.set_ylabel("Count")
        ax.set_title("Rotation Error — SITR [tactile]")
        ax.legend()

        ax = axes[1]
        ax.hist(trans_err, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.axvline(trans_err.mean(), color="red", ls="--",
                   label=f"Mean: {trans_err.mean():.3f}")
        ax.axvline(np.median(trans_err), color="orange", ls="--",
                   label=f"Median: {np.median(trans_err):.3f}")
        ax.set_xlabel("Translation Error (L1, normalized)")
        ax.set_ylabel("Count")
        ax.set_title("Translation Error — SITR [tactile]")
        ax.legend()

        plt.tight_layout()
        hist_path = osp.join(args.save_path, "pose_error_hist.png")
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"Pose histogram saved -> {hist_path}")

    # ── Real data eval (qualitative) ────────────────────────────────────────
    if args.real_dir:
        calib_from = args.calib_from or osp.join(
            args.data_path, "edge", "session_000", "sensor_0000")
        encoder_raw = model.encoder.sitr
        real_save = osp.join(args.save_path, "real")
        eval_real(model, encoder_raw, pose_head, args.real_dir, calib_from,
                  args.num_real, device, real_save,
                  raw_input=args.raw_input,
                  calibration_config=args.calibration_config,
                  depth_from_npy=args.depth_from_npy,
                  center_crop=args.center_crop)


if __name__ == "__main__":
    main()
