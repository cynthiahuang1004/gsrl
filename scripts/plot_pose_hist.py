"""Plot histogram of per-sample pose rotation error on the val set — SITR & DINOv3."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.amp import autocast
from torch.utils.data import DataLoader

from train_pose_sitr import PoseHead, PoseDataset, AverageMeter
from train_pose_dinov3 import DINOv3Encoder
from models.networks import SITR_base
from dataloaders import imagenet_mu, imagenet_std, sample_mu, sample_std
import torchvision.transforms as T


MODELS = {
    "sitr": {
        "label": "SITR (calib=0, finetuned)",
        "color": "#4C72B0",
        "encoder_weights": "output_checkpoints/20260708_sitr_finetune/best.pth",
        "pose_weights": "output_checkpoints/20260708_pose_sitr/best.pth",
        "encoder_type": "sitr",
        "calibration_config": 0,
    },
    "dinov3": {
        "label": "DINOv3 ViT-L/16",
        "color": "#DD8452",
        "dinov3_weights": "output_checkpoints/dpt_dinov3/dinov3_vitl16_pretrain_lvd1689m.pth",
        "pose_weights": "output_checkpoints/20260709_pose_dinov3/best.pth",
        "encoder_type": "dinov3",
        "calibration_config": 0,
    },
}

DATA_PATH = "/media/hdd2/ihsuan/gs_blender/renders"
MESH_DIR = "/media/hdd2/ihsuan/gs_blender/meshes"
OUT_DIR = "eval_results/pose_comparison"


def build_encoder_and_head(cfg, device):
    pose_ck = torch.load(cfg["pose_weights"], map_location="cpu", weights_only=False)
    pose_args = pose_ck.get("args", {})

    if cfg["encoder_type"] == "sitr":
        encoder = SITR_base(num_calibration=cfg["calibration_config"]).to(device)
        state = torch.load(cfg["encoder_weights"], map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        encoder.load_state_dict(state, strict=False)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        embed_dim = 768
    else:
        encoder = DINOv3Encoder(
            model_name="dinov3_vitl16", weights=cfg["dinov3_weights"]
        ).to(device)
        encoder.eval()
        embed_dim = encoder.embed_dim

    pose_head = PoseHead(
        dim=embed_dim,
        hidden_dim=pose_args.get("hidden_dim", 256),
        dropout=0.0,
        pose_mode=pose_args.get("pose_mode", "regression"),
        rot_num_bins=pose_args.get("rot_num_bins", 72),
    ).to(device)
    pose_head.load_state_dict(pose_ck["pose_head"])
    pose_head.eval()

    return encoder, pose_head


def run_encoder(encoder, encoder_type, imgs, calibs):
    if encoder_type == "sitr":
        latent = encoder.forward_encoder(imgs, calibs)
    else:
        latent = encoder(imgs)["latent"]
    return latent[:, 0, :], latent[:, 1:, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}

    for name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {cfg['label']}")
        print(f"{'='*60}")

        if cfg["calibration_config"] == 0:
            img_xform = T.Compose([T.ToTensor(), T.Normalize(mean=imagenet_mu, std=imagenet_std)])
        else:
            img_xform = T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)])

        val_ds = PoseDataset(
            DATA_PATH, MESH_DIR, img_xform,
            calibration_config=cfg["calibration_config"],
            split="val", val_every=20,
            raw_input=(cfg["calibration_config"] == 0),
            center_crop=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

        encoder, pose_head = build_encoder_and_head(cfg, device)

        all_rot_deg = []
        all_trans_err = []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["sample"].to(device)
                calibs = batch["calibration"].to(device)
                gt_pose = batch["pose"].to(device)

                with autocast("cuda"):
                    cls_tok, spatial = run_encoder(encoder, cfg["encoder_type"], imgs, calibs)
                    pred = pose_head(cls_tok, spatial)

                se2 = pred["se2"].float()
                cos_p, sin_p = se2[:, 0], se2[:, 1]
                cos_g, sin_g = gt_pose[:, 0], gt_pose[:, 1]

                dot = (cos_p * cos_g + sin_p * sin_g).clamp(-1 + 1e-6, 1 - 1e-6)
                rot_deg = torch.acos(dot) * 180.0 / math.pi
                all_rot_deg.append(rot_deg.cpu().numpy())

                trans_err = (se2[:, 2:] - gt_pose[:, 2:]).abs().sum(dim=-1)
                all_trans_err.append(trans_err.cpu().numpy())

        rot_deg = np.concatenate(all_rot_deg)
        trans_err = np.concatenate(all_trans_err)
        results[name] = {"rot_deg": rot_deg, "trans_err": trans_err,
                         "label": cfg["label"], "color": cfg["color"]}

        print(f"  Samples: {len(rot_deg)}")
        print(f"  Rotation (deg): mean={rot_deg.mean():.2f}, median={np.median(rot_deg):.2f}, "
              f"std={rot_deg.std():.2f}, max={rot_deg.max():.2f}")
        print(f"  Trans (L1): mean={trans_err.mean():.4f}, median={np.median(trans_err):.4f}")

        del encoder, pose_head
        torch.cuda.empty_cache()

    # --- Plot: side-by-side comparison ---
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(8 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, (name, r) in enumerate(results.items()):
        rot_deg = r["rot_deg"]
        trans_err = r["trans_err"]
        color = r["color"]
        label = r["label"]

        ax = axes[0, i]
        ax.hist(rot_deg, bins=60, range=(0, 180), color=color,
                edgecolor="white", alpha=0.85)
        ax.axvline(rot_deg.mean(), color="red", ls="--",
                   label=f"Mean: {rot_deg.mean():.1f}°")
        ax.axvline(np.median(rot_deg), color="orange", ls="--",
                   label=f"Median: {np.median(rot_deg):.1f}°")
        ax.set_xlabel("Rotation Error (degrees)")
        ax.set_ylabel("Count")
        ax.set_title(f"Rotation — {label}")
        ax.legend()

        ax = axes[1, i]
        ax.hist(trans_err, bins=60, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(trans_err.mean(), color="red", ls="--",
                   label=f"Mean: {trans_err.mean():.3f}")
        ax.axvline(np.median(trans_err), color="orange", ls="--",
                   label=f"Median: {np.median(trans_err):.3f}")
        ax.set_xlabel("Translation Error (L1, normalized)")
        ax.set_ylabel("Count")
        ax.set_title(f"Translation — {label}")
        ax.legend()

    plt.suptitle("Pose Error Distribution (val set) — SITR vs DINOv3", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(args.out_dir, "pose_error_hist.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # --- Overlaid rotation histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, r in results.items():
        ax.hist(r["rot_deg"], bins=60, range=(0, 180), color=r["color"],
                edgecolor="white", alpha=0.5, label=f'{r["label"]} (mean={r["rot_deg"].mean():.1f}°)')
    ax.set_xlabel("Rotation Error (degrees)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Rotation Error Comparison — SITR vs DINOv3", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    overlay_path = os.path.join(args.out_dir, "pose_rot_overlay.png")
    fig.savefig(overlay_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {overlay_path}")


if __name__ == "__main__":
    main()
