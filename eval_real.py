"""
eval_real.py — Run trained SITR / DPT models on REAL GelSlim tactile images
to qualitatively check sim-to-real transfer.

Pipeline matches sim training preprocessing:
  real_sample = real_tactile - base_image   (raw pixel diff, NOT /255)
  -> ToTensor (no scaling for float) -> Normalize(sample_mu, sample_std)

Calibration: real data has no SITR calibration probes, so we borrow a SIM
sensor's 18 calibration images as a stand-in (this mismatch is itself part of
the sim-to-real gap — flagged in the output).

Usage:
  python eval_real.py \
    --tactile-dir /media/hdd/ihsuan/gelslim_depth/datasets_new/pattern_01_2_lines_angles_1/tactile_image \
    --base-image  /media/hdd/ihsuan/gelslim_depth/datasets_new/base_tactile_image.jpg \
    --calib-from  /media/hdd/ihsuan/gs_blender/renders/edge/session_000/sensor_0000 \
    --sitr-weights   output_checkpoints/sitr_finetune_gsblender_gtnorm/best.pth \
    --dpt-encoder    output_checkpoints/sitr_finetune_gsblender_gtnorm/best.pth \
    --dpt-decoder    output_checkpoints/dpt_stage2_gtnorm_unfreeze4/best.pth \
    --out eval_results/real_pattern01 --num 8 --device cuda:1
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataloaders import sample_mu, sample_std, norm_mu, norm_std, dmap_mu, dmap_std, imagenet_mu, imagenet_std, raw_mu, raw_std
from models.networks import SITR_base
from models.dpt import SITRWithDPT, DINOv2WithDPT, DINOv3WithDPT

IMG = 224
CALIB_LIST = list(range(1, 19))  # calibration_config = 18


def load_resized(path, size=IMG):
    """Load an image as float32 HWC array resized to size x size (RGB)."""
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def make_transform():
    return T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)])


def build_calib(calib_from, tf):
    """Build the (1, 54, 224, 224) calibration tensor from a sim sensor dir."""
    cal_dir = os.path.join(calib_from, "calibration")
    ref = load_resized(os.path.join(cal_dir, "0000.png"))
    chans = []
    for i in CALIB_LIST:
        c = load_resized(os.path.join(cal_dir, "{0:04}.png".format(i)))
        chans.append(tf(c - ref))           # (3,224,224)
    calib = torch.cat(chans, dim=0)          # (54,224,224)
    return calib.unsqueeze(0)


def build_calib_raw(calib_from, tf_raw):
    """Build (1, 57, 224, 224) calibration tensor for raw input mode (config=19)."""
    cal_dir = os.path.join(calib_from, "calibration")
    chans = []
    for i in range(19):
        c = load_resized(os.path.join(cal_dir, "{0:04}.png".format(i)))
        chans.append(tf_raw(c))             # (3,224,224)
    calib = torch.cat(chans, dim=0)          # (57,224,224)
    return calib.unsqueeze(0)


def unnorm_normal(t):
    """(3,H,W) normalized -> (H,W,3) uint8."""
    mu = torch.tensor(norm_mu).view(3, 1, 1)
    sd = torch.tensor(norm_std).view(3, 1, 1)
    img = (t * sd + mu).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
    return img


def unnorm_depth(t):
    """(1,H,W) normalized -> (H,W) float."""
    return (t.squeeze(0) * dmap_std[0] + dmap_mu[0]).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tactile-dir", required=True)
    p.add_argument("--base-image", required=True)
    p.add_argument("--calib-from", required=True,
                   help="sim sensor dir containing calibration/ (stand-in)")
    p.add_argument("--sitr-weights", default=None)
    p.add_argument("--dpt-encoder", default=None)
    p.add_argument("--dpt-decoder", default=None)
    p.add_argument("--dinov2-decoder", default=None,
                   help="DINOv2+DPT decoder checkpoint (uses raw images, ImageNet norm)")
    p.add_argument("--dinov2-model", default="dinov2_vitb14")
    p.add_argument("--dinov3-decoder", default=None,
                   help="DINOv3+DPT decoder checkpoint")
    p.add_argument("--dinov3-model", default="dinov3_vitl16")
    p.add_argument("--dinov3-weights", default=None,
                   help="Local .pth path for DINOv3 pretrained backbone weights")
    p.add_argument("--raw-input", action="store_true", default=False)
    p.add_argument("--calibration-config", type=int, default=18)
    p.add_argument("--out", default="eval_results/real")
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.raw_input and args.calibration_config == 18:
        args.calibration_config = 19

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tf = make_transform()
    num_calib = args.calibration_config

    # ── inputs ──────────────────────────────────────────────────────────
    base = load_resized(args.base_image)
    files = sorted(glob.glob(os.path.join(args.tactile_dir, "*.jpg")))
    step = max(1, len(files) // args.num)
    files = files[::step][:args.num]
    print(f"{len(files)} real samples; base={args.base_image}")

    raws = [load_resized(f).astype(np.uint8) for f in files]
    diffs = [load_resized(f) - base for f in files]

    if args.raw_input:
        tf_raw = T.Compose([T.ToTensor(), T.Normalize(mean=raw_mu, std=raw_std)])
        samples = torch.stack([tf_raw(load_resized(f)) for f in files]).to(device)
        calib = build_calib_raw(args.calib_from, tf_raw).repeat(len(files), 1, 1, 1).to(device)
    else:
        samples = torch.stack([tf(load_resized(f) - base) for f in files]).to(device)
        calib = build_calib(args.calib_from, tf).repeat(len(files), 1, 1, 1).to(device)
    print(f"sample batch {tuple(samples.shape)}, calib {tuple(calib.shape)} (stand-in from {args.calib_from})")

    # ── SITR ────────────────────────────────────────────────────────────
    sitr_normal = None
    if args.sitr_weights:
        m = SITR_base(num_calibration=num_calib)
        ck = torch.load(args.sitr_weights, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        m = m.to(device).eval()
        with torch.no_grad():
            sitr_normal = m(samples, calib)["proj"].cpu()
        del m
        print("SITR done")

    # ── DPT ─────────────────────────────────────────────────────────────
    dpt_depth = dpt_normal = None
    if args.dpt_decoder and args.dpt_encoder:
        sitr = SITR_base(num_calibration=num_calib)
        ck = torch.load(args.dpt_encoder, map_location="cpu", weights_only=False)
        sitr.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        model = SITRWithDPT(sitr, embed_dim=768, features=256).to(device)
        dck = torch.load(args.dpt_decoder, map_location="cpu", weights_only=False)
        model.decoder.load_state_dict(dck["decoder"] if "decoder" in dck else dck)
        if "encoder" in dck:
            model.encoder.sitr.load_state_dict(dck["encoder"])
            print("  (DPT ckpt has finetuned encoder — loaded)")
        model.eval()
        with torch.no_grad():
            out = model(samples, calib)
        dpt_depth = out["depth"].cpu()
        dpt_normal = out["normal"].cpu()
        del model, sitr
        print("DPT done")

    # ── DINOv2 ──────────────────────────────────────────────────────────
    dino_depth = dino_normal = None
    if args.dinov2_decoder:
        tf_inet = T.Compose([T.ToTensor(), T.Normalize(mean=imagenet_mu, std=imagenet_std)])
        raw_samples = torch.stack([tf_inet(load_resized(f)) for f in files]).to(device)
        model = DINOv2WithDPT(model_name=args.dinov2_model, features=256).to(device)
        dck = torch.load(args.dinov2_decoder, map_location="cpu", weights_only=False)
        model.decoder.load_state_dict(dck["decoder"] if "decoder" in dck else dck)
        model.eval()
        with torch.no_grad():
            out = model(raw_samples)
        dino_depth = out["depth"].cpu()
        dino_normal = out["normal"].cpu()
        del model, raw_samples
        print("DINOv2 done")

    # ── DINOv3 ──────────────────────────────────────────────────────────
    dinov3_depth = dinov3_normal = None
    if args.dinov3_decoder:
        tf_inet = T.Compose([T.ToTensor(), T.Normalize(mean=imagenet_mu, std=imagenet_std)])
        raw_samples = torch.stack([tf_inet(load_resized(f)) for f in files]).to(device)
        model = DINOv3WithDPT(
            model_name=args.dinov3_model,
            weights=args.dinov3_weights,
            features=256,
        ).to(device)
        dck = torch.load(args.dinov3_decoder, map_location="cpu", weights_only=False)
        model.decoder.load_state_dict(dck["decoder"] if "decoder" in dck else dck)
        model.eval()
        with torch.no_grad():
            out = model(raw_samples)
        dinov3_depth = out["depth"].cpu()
        dinov3_normal = out["normal"].cpu()
        del model, raw_samples
        print("DINOv3 done")

    # ── visualize ───────────────────────────────────────────────────────
    ncol = 2 + (1 if sitr_normal is not None else 0) \
             + (2 if dpt_depth is not None else 0) \
             + (2 if dino_depth is not None else 0) \
             + (2 if dinov3_depth is not None else 0)
    for i, f in enumerate(files):
        fig, axes = plt.subplots(1, ncol, figsize=(3.4 * ncol, 3.6))
        c = 0
        axes[c].imshow(raws[i]); axes[c].set_title(f"Real raw\n{os.path.basename(f)}", fontsize=9); c += 1
        d = diffs[i]; dv = np.clip((d - d.min()) / (np.ptp(d) + 1e-6) * 255, 0, 255).astype(np.uint8)
        axes[c].imshow(dv); axes[c].set_title("Bg-subtracted\n(model input)", fontsize=9); c += 1
        if sitr_normal is not None:
            axes[c].imshow(unnorm_normal(sitr_normal[i])); axes[c].set_title("SITR normal", fontsize=9); c += 1
        if dpt_depth is not None:
            axes[c].imshow(unnorm_normal(dpt_normal[i])); axes[c].set_title("DPT normal", fontsize=9); c += 1
            im = axes[c].imshow(unnorm_depth(dpt_depth[i]), cmap="viridis")
            axes[c].set_title("DPT depth", fontsize=9)
            fig.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04); c += 1
        if dino_depth is not None:
            axes[c].imshow(unnorm_normal(dino_normal[i])); axes[c].set_title("DINOv2 normal", fontsize=9); c += 1
            im = axes[c].imshow(unnorm_depth(dino_depth[i]), cmap="viridis")
            axes[c].set_title("DINOv2 depth", fontsize=9)
            fig.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04); c += 1
        if dinov3_depth is not None:
            axes[c].imshow(unnorm_normal(dinov3_normal[i])); axes[c].set_title("DINOv3 normal", fontsize=9); c += 1
            im = axes[c].imshow(unnorm_depth(dinov3_depth[i]), cmap="viridis")
            axes[c].set_title("DINOv3 depth", fontsize=9)
            fig.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04); c += 1
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"real_{i:03d}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"saved {len(files)} figures -> {args.out}/")


if __name__ == "__main__":
    main()
