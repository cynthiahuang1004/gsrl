"""
eval_dinov3_pipeline.py — Evaluate DINOv3 encoder + DPT decoder + Pose head
Metrics matching VisTacFusion format.

Usage:
  python eval_dinov3_pipeline.py \
    --dinov3-weights output_checkpoints/dpt_dinov3/dinov3_vitl16_pretrain.pth \
    --dpt-weights output_checkpoints/20260709_dpt_dinov3/best.pth \
    --pose-weights output_checkpoints/20260709_pose_dinov3/best.pth \
    --save-path eval_results/20260710_dinov3_pipeline \
    --device cuda:2 --raw-input --center-crop --depth-from-npy
"""
import argparse
import os
import os.path as osp

import json
import math
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

from models.dpt import DINOv3WithDPT
from train_pose_dinov3 import DINOv3Encoder
from train_pose_sitr import PoseHead, PoseDataset
from dataloaders import (sim_dataset_nested, sample_mu, sample_std,
                         norm_mu, norm_std, dmap_mu, dmap_std,
                         imagenet_mu, imagenet_std)
from eval_sitr_pipeline import (_angular_error_deg, _depth_metrics,
                                _rotation_error_deg, unnorm)
import torchvision.transforms as T


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="/media/hdd2/ihsuan/gs_blender/renders")
    p.add_argument("--mesh-dir", default="/media/hdd2/ihsuan/gs_blender/meshes")
    p.add_argument("--dinov3-model", default="dinov3_vitl16")
    p.add_argument("--dinov3-weights",
                   default="output_checkpoints/dpt_dinov3/dinov3_vitl16_pretrain_lvd1689m.pth")
    p.add_argument("--dpt-weights", required=True)
    p.add_argument("--pose-weights", default=None)
    p.add_argument("--val-every", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save-path", default=None,
                   help="Output dir. Default: eval_results/<dpt-weights folder name>")
    p.add_argument("--center-crop", action="store_true", default=False)
    p.add_argument("--raw-input", action="store_true", default=False)
    p.add_argument("--depth-from-npy", action="store_true", default=False)
    p.add_argument("--real-dir", default=None)
    p.add_argument("--num-real", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.save_path is None:
        ckpt_dir_name = osp.basename(osp.dirname(args.dpt_weights))
        args.save_path = osp.join("eval_results", ckpt_dir_name)
    os.makedirs(args.save_path, exist_ok=True)

    if args.raw_input:
        img_xform = T.Compose([T.ToTensor(), T.Normalize(mean=imagenet_mu, std=imagenet_std)])
    else:
        img_xform = T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)])
    dmap_xform = T.Compose([T.ToTensor(), T.Normalize(mean=dmap_mu, std=dmap_std)])
    norm_xform = T.Compose([T.ToTensor(), T.Normalize(mean=norm_mu, std=norm_std)])

    # ── Dataset ────────────────────────────────────────────────────────────
    print("Loading val dataset...")
    full_ds = sim_dataset_nested(
        path=args.data_path, augment=False,
        transforms=img_xform, dmap_transforms=dmap_xform, norm_transforms=norm_xform,
        calibration_config=0, sendTwo=False,
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

    # ── DINOv3 + DPT model ─────────────────────────────────────────────────
    print(f"Building DINOv3 encoder ({args.dinov3_model})...")
    model = DINOv3WithDPT(
        model_name=args.dinov3_model, weights=args.dinov3_weights,
        features=256, dropout=0.0,
    ).to(device)
    print(f"Loading DPT decoder from {args.dpt_weights}...")
    dpt_ck = torch.load(args.dpt_weights, map_location="cpu", weights_only=False)
    model.decoder.load_state_dict(dpt_ck["decoder"])
    model.eval()

    # ── Pose encoder + head ────────────────────────────────────────────────
    pose_head = None
    pose_encoder = None
    obj_embedding = None
    if args.pose_weights:
        print(f"Loading pose head from {args.pose_weights}...")
        pose_ck = torch.load(args.pose_weights, map_location="cpu", weights_only=False)
        pose_args = pose_ck.get("args", {})
        embed_dim = model.encoder.embed_dim

        pose_encoder = DINOv3Encoder(
            model_name=args.dinov3_model, weights=args.dinov3_weights
        ).to(device)
        pose_encoder.eval()

        pose_head = PoseHead(
            dim=embed_dim,
            hidden_dim=pose_args.get("hidden_dim", 256),
            dropout=0.0,
            pose_mode=pose_args.get("pose_mode", "regression"),
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

    # ── Vis indices (one per object) ───────────────────────────────────────
    obj_vis_indices = {}
    for vi in val_idx:
        unit_idx = vi // spu
        unit = full_ds.units[unit_idx]
        obj = osp.basename(osp.dirname(osp.dirname(unit)))
        if obj not in obj_vis_indices:
            obj_vis_indices[obj] = vi
    vis_global_set = set(obj_vis_indices.values())
    print(f"  Will visualize {len(vis_global_set)} samples (1 per object)")

    # ── Pose GT (lightweight — only reads pose json, no images) ─────────────
    pose_gt_map = {}
    pose_obj_map = {}
    if pose_head is not None:
        pose_ds = PoseDataset(
            args.data_path, args.mesh_dir, img_xform,
            calibration_config=0, raw_input=args.raw_input,
            split="val", val_every=args.val_every,
            center_crop=args.center_crop)
        for pi in range(len(pose_ds)):
            unit, sample_idx = pose_ds.samples[pi]
            meta = pose_ds.unit_meta[unit]
            with open(osp.join(unit, "raw_data", f"{sample_idx:04d}_pose.json")) as f:
                pdata = json.load(f)
            delta_rz = pdata["rotation_euler"][2] - meta["rz0"]
            half = meta["half"]
            cos_rz, sin_rz = math.cos(delta_rz), math.sin(delta_rz)
            sx, sy = pdata["sample_x"], pdata["sample_y"]
            x_norm = (cos_rz * sx - sin_rz * sy) / max(half, 1e-8)
            y_norm = (sin_rz * sx + cos_rz * sy) / max(half, 1e-8)
            pose_gt_map[pi] = np.array([cos_rz, sin_rz, x_norm, y_norm], dtype=np.float32)
            if obj_embedding is not None:
                obj_name = osp.basename(osp.dirname(osp.dirname(unit)))
                pose_obj_map[pi] = pose_ds._obj_to_id[obj_name]

    # ── Eval loop ──────────────────────────────────────────────────────────
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
        imgs = batch["sample"].to(device)
        gt_depth_t = batch["dmap"]
        gt_norm_t = batch["norm"]
        B = imgs.size(0)

        with torch.no_grad(), autocast("cuda"):
            out = model(imgs)

            pred_se2 = None
            if pose_head is not None and pose_encoder is not None:
                enc_out = pose_encoder(imgs)
                latent = enc_out["latent"]
                if obj_embedding is not None:
                    obj_ids = torch.tensor(
                        [pose_obj_map[sample_counter + j] for j in range(B)],
                        device=device)
                    latent = latent + obj_embedding(obj_ids).unsqueeze(1)
                pred_pose = pose_head(latent[:, 0, :], latent[:, 1:, :])
                pred_se2 = pred_pose["se2"].float().cpu().numpy()

        pred_depth_t = out["depth"].cpu()
        pred_norm_t = out["normal"].cpu()

        if args.depth_from_npy:
            pred_d, gt_d = pred_depth_t, gt_depth_t
            pred_n, gt_n = pred_norm_t, gt_norm_t
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
            metrics["normal_mse"].append(
                F.mse_loss(torch.from_numpy(pn).float(),
                           torch.from_numpy(gn).float()).item())
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

    # ── Report ─────────────────────────────────────────────────────────────
    summary = {k: float(np.mean(v)) for k, v in metrics.items() if v}

    print("\n" + "=" * 60)
    print("DINOv3 Pipeline Evaluation Results")
    print("=" * 60)

    report_path = osp.join(args.save_path, "eval_metrics.txt")
    with open(report_path, "w") as f:
        f.write("DINOv3 Pipeline Evaluation Results\n")
        f.write(f"Encoder: {args.dinov3_model} ({args.dinov3_weights})\n")
        f.write(f"DPT: {args.dpt_weights}\n")
        if args.pose_weights:
            f.write(f"Pose: {args.pose_weights}\n")
        f.write(f"Val samples: {len(val_ds)}\n\n")

        f.write("[tactile]\n")
        for key, label in [
            ("depth_mse", "Depth MSE"),
            ("depth_mae", "Depth MAE"),
            ("depth_rmse", "Depth RMSE"),
            ("depth_d1", "Depth delta<1.25 (%)"),
            ("normal_mse", "Normal MSE"),
            ("normal_ang_mean", "Normal Ang Mean (deg)"),
            ("normal_ang_median", "Normal Ang Median (deg)"),
            ("pose_rot_deg", "Pose Rot Error (deg)"),
            ("pose_rot_loss", "Pose Rot Loss (1-cos)"),
            ("pose_rot_l1", "Pose Rot L1 (cos/sin)"),
            ("pose_trans_l1", "Pose Trans L1"),
        ]:
            fmt = ".2f" if "deg" in key or "d1" in key else ".6f"
            f.write(f"  {label:27s}{summary.get(key, float('nan')):{fmt}}\n")

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
        ax.hist(rot_deg, bins=60, range=(0, 180), color="#DD8452",
                edgecolor="white", alpha=0.85)
        ax.axvline(rot_deg.mean(), color="red", ls="--",
                   label=f"Mean: {rot_deg.mean():.1f}°")
        ax.axvline(np.median(rot_deg), color="orange", ls="--",
                   label=f"Median: {np.median(rot_deg):.1f}°")
        ax.set_xlabel("Rotation Error (degrees)")
        ax.set_ylabel("Count")
        ax.set_title("Rotation Error — DINOv3 [tactile]")
        ax.legend()

        ax = axes[1]
        ax.hist(trans_err, bins=60, color="#DD8452", edgecolor="white", alpha=0.85)
        ax.axvline(trans_err.mean(), color="red", ls="--",
                   label=f"Mean: {trans_err.mean():.3f}")
        ax.axvline(np.median(trans_err), color="orange", ls="--",
                   label=f"Median: {np.median(trans_err):.3f}")
        ax.set_xlabel("Translation Error (L1, normalized)")
        ax.set_ylabel("Count")
        ax.set_title("Translation Error — DINOv3 [tactile]")
        ax.legend()

        plt.tight_layout()
        hist_path = osp.join(args.save_path, "pose_error_hist.png")
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"Pose histogram saved -> {hist_path}")

    # ── Real data (qualitative) ────────────────────────────────────────────
    if args.real_dir:
        from eval_sitr_pipeline import eval_real
        from dataloaders import fixed_center_crop
        encoder_raw = pose_encoder if pose_encoder is not None else None
        real_save = osp.join(args.save_path, "real")
        eval_real(model, encoder_raw, pose_head, args.real_dir, None,
                  args.num_real, device, real_save,
                  raw_input=args.raw_input,
                  calibration_config=0,
                  depth_from_npy=args.depth_from_npy,
                  center_crop=args.center_crop)


if __name__ == "__main__":
    main()
