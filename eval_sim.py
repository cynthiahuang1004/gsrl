"""
eval_sim.py – Evaluate SITR finetune & DPT Stage 2 models on simulation data

Produces:
  1. Side-by-side visualizations (input / prediction / GT)
  2. Quantitative metrics (MSE, MAE, angular error for normals, etc.)

Usage
-----
# Evaluate SITR finetune (normal reconstruction)
python3 eval_sim.py \
    --mode sitr \
    --data-path /media/hdd/ihsuan/gs_blender/renders \
    --weights output_checkpoints/sitr_finetune_gsblender_gtnorm/best.pth \
    --val-objects edge hex_key pattern_31_rod \
    --gt-norm
    --save-path eval_results/sitr_finetune_gtnorm \
    --num-vis 20

# Evaluate DPT (depth + normal)
python3 eval_sim.py \
    --mode dpt \
    --data-path /media/hdd/ihsuan/gs_blender/renders \
    --encoder-weights output_checkpoints/sitr_finetune_gsblender_gtnorm/best.pth \
    --decoder-weights output_checkpoints/dpt_stage2_gsblender_gtnorm/best.pth \
    --val-objects edge hex_key pattern_31_rod \
    --gt-norm \
    --save-path eval_results/dpt_stage2_gtnorm \
    --num-vis 20
"""

import argparse
import os
import math

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torchvision.transforms as T
from dataloaders import sim_dataset_nested, norm_mu, norm_std, dmap_mu, dmap_std, sample_mu, sample_std, raw_mu, raw_std, imagenet_mu, imagenet_std
from models.networks import SITR_base
from models.dpt import SITRWithDPT, DINOv2WithDPT, DINOv3WithDPT, DAv2WithDPT


# ── unnormalize helpers ──────────────────────────────────────────────────────

def unnorm_image(tensor, mu, std):
    """(C,H,W) normalized tensor -> (H,W,C) uint8 numpy for display."""
    mu = torch.tensor(mu, dtype=tensor.dtype).view(-1, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    img = tensor * std + mu
    img = img.permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)
    return img


def unnorm_norm(tensor):
    """Unnormalize a normal map tensor (3,H,W) -> (H,W,3) uint8."""
    return unnorm_image(tensor, norm_mu, norm_std)


def unnorm_dmap(tensor):
    """Unnormalize a depth map tensor (1,H,W) -> (H,W) float."""
    mu = torch.tensor(dmap_mu, dtype=tensor.dtype).view(-1, 1, 1)
    std = torch.tensor(dmap_std, dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor * std + mu).squeeze(0).numpy()


def unnorm_sample(tensor):
    """Unnormalize a tactile image tensor (3,H,W) -> (H,W,3) uint8."""
    return unnorm_image(tensor, sample_mu, sample_std)


def unnorm_raw(tensor):
    return unnorm_image(tensor, raw_mu, raw_std)


def unnorm_imagenet(tensor):
    return unnorm_image(tensor, imagenet_mu, imagenet_std)


# ── metrics ──────────────────────────────────────────────────────────────────

def angular_error_deg(pred_norm, gt_norm):
    """Mean angular error in degrees between predicted and GT normals.
    Both inputs: (B, 3, H, W), unnormalized pixel values [0,255]."""
    p = pred_norm.float() / 255.0 * 2.0 - 1.0
    g = gt_norm.float() / 255.0 * 2.0 - 1.0

    p = F.normalize(p, dim=1)
    g = F.normalize(g, dim=1)

    cos_sim = (p * g).sum(dim=1).clamp(-1, 1)
    angle = torch.acos(cos_sim) * (180.0 / math.pi)
    return angle.mean().item(), angle.median().item()


def depth_metrics(pred, gt):
    """Compute depth metrics. Both inputs: (B, 1, H, W) unnormalized."""
    mask = gt != 0
    if mask.sum() == 0:
        return {}
    p = pred[mask].float()
    g = gt[mask].float()

    abs_diff = (p - g).abs()
    mae = abs_diff.mean().item()
    mse = (abs_diff ** 2).mean().item()
    rmse = mse ** 0.5

    ratio = torch.max(p / g.clamp(min=1e-6), g / p.clamp(min=1e-6))
    d1 = (ratio < 1.25).float().mean().item() * 100
    d2 = (ratio < 1.25 ** 2).float().mean().item() * 100

    return {"MAE": mae, "RMSE": rmse, "delta<1.25": d1, "delta<1.56": d2}


# ── visualization ────────────────────────────────────────────────────────────

def _obj_name_from_dataset(ds, global_idx):
    """Extract object name from a sim_dataset_nested given a global sample index."""
    unit_idx = global_idx // ds.samples_per_unit
    unit = ds.units[unit_idx]
    return os.path.basename(os.path.dirname(os.path.dirname(unit)))


def _pick_vis_indices(dataset, num_vis):
    """Pick evenly-spaced indices across objects so each object is represented."""
    objs = dataset.objects
    spu = dataset.samples_per_unit
    units = dataset.units

    obj_units = {o: [] for o in objs}
    for ui, u in enumerate(units):
        o = os.path.basename(os.path.dirname(os.path.dirname(u)))
        obj_units[o].append(ui)

    per_obj = max(1, num_vis // len(objs))
    chosen = []
    for o in objs:
        uis = obj_units[o]
        step = max(1, len(uis) // per_obj)
        picked = 0
        for j in range(0, len(uis), step):
            if picked >= per_obj:
                break
            chosen.append(uis[j] * spu)
            picked += 1

    return set(chosen[:num_vis])


def vis_sitr(sample, pred_norm, gt_norm, save_path, idx, obj_name=""):
    """Visualize SITR: input | predicted normal | GT normal."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    inp = unnorm_sample(sample)
    inp = np.clip(inp, 0, 255).astype(np.uint8)
    axes[0].imshow(inp)
    axes[0].set_title(f"Input [{obj_name}]")

    pn = unnorm_norm(pred_norm)
    axes[1].imshow(pn)
    axes[1].set_title("Predicted Normal")

    gn = unnorm_norm(gt_norm)
    axes[2].imshow(gn)
    axes[2].set_title("GT Normal")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    tag = f"_{obj_name}" if obj_name else ""
    fig.savefig(os.path.join(save_path, f"sitr_{idx:04d}{tag}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def vis_dpt(sample, pred_depth, gt_depth, pred_norm, gt_norm, save_path, idx, obj_name="", unnorm_fn=None):
    """Visualize DPT: input | pred depth | GT depth | depth error | pred normal | GT normal."""
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))

    _unnorm = unnorm_fn or unnorm_sample
    inp = _unnorm(sample)
    inp = np.clip(inp, 0, 255).astype(np.uint8)
    axes[0].imshow(inp)
    axes[0].set_title(f"Input [{obj_name}]")

    pd = unnorm_dmap(pred_depth)
    gd = unnorm_dmap(gt_depth)
    vmin = min(pd.min(), gd.min())
    vmax = max(pd.max(), gd.max())

    im1 = axes[1].imshow(pd, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Pred Depth")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(gd, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[2].set_title("GT Depth")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    err = np.abs(pd - gd)
    im3 = axes[3].imshow(err, cmap="hot")
    axes[3].set_title(f"Depth Err (MAE={err.mean():.1f})")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    pn = unnorm_norm(pred_norm)
    axes[4].imshow(pn)
    axes[4].set_title("Pred Normal")

    gn = unnorm_norm(gt_norm)
    axes[5].imshow(gn)
    axes[5].set_title("GT Normal")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    tag = f"_{obj_name}" if obj_name else ""
    fig.savefig(os.path.join(save_path, f"dpt_{idx:04d}{tag}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── evaluation loops ─────────────────────────────────────────────────────────

@torch.no_grad()
def eval_sitr(model, loader, device, save_path, num_vis):
    model.eval()
    all_mse = []
    all_ang_mean = []
    all_ang_median = []
    vis_count = 0

    ds = loader.dataset
    vis_indices = _pick_vis_indices(ds, num_vis) if num_vis > 0 else set()

    for batch in tqdm(loader, desc="Evaluating SITR"):
        imgs = batch["sample"].to(device)
        calibs = batch["calibration"].to(device)
        gt_norm_b = batch["norm"].to(device)
        idxs = batch["idx"]

        out = model(imgs, calibs)
        pred_norm = out["proj"]

        mse = F.mse_loss(pred_norm, gt_norm_b).item()
        all_mse.append(mse)

        pred_unnorm = pred_norm.cpu() * torch.tensor(norm_std).view(1, 3, 1, 1) + torch.tensor(norm_mu).view(1, 3, 1, 1)
        gt_unnorm = gt_norm_b.cpu() * torch.tensor(norm_std).view(1, 3, 1, 1) + torch.tensor(norm_mu).view(1, 3, 1, 1)
        ang_mean, ang_median = angular_error_deg(pred_unnorm, gt_unnorm)
        all_ang_mean.append(ang_mean)
        all_ang_median.append(ang_median)

        if vis_count < num_vis:
            for i in range(imgs.size(0)):
                gidx = idxs[i].item()
                if gidx in vis_indices:
                    obj_name = _obj_name_from_dataset(ds, gidx)
                    vis_sitr(imgs[i].cpu(), pred_norm[i].cpu(), gt_norm_b[i].cpu(),
                             save_path, vis_count, obj_name)
                    vis_count += 1

    metrics = {
        "Normal MSE (normalized)": np.mean(all_mse),
        "Angular Error Mean (deg)": np.mean(all_ang_mean),
        "Angular Error Median (deg)": np.mean(all_ang_median),
    }
    return metrics


@torch.no_grad()
def eval_dpt(model, loader, device, save_path, num_vis, unnorm_fn=None):
    model.eval()
    all_norm_mse = []
    all_depth_mse = []
    all_ang_mean = []
    all_depth_mae = []
    vis_count = 0

    ds = loader.dataset
    vis_indices = _pick_vis_indices(ds, num_vis) if num_vis > 0 else set()

    for batch in tqdm(loader, desc="Evaluating DPT"):
        imgs = batch["sample"].to(device)
        calibs = batch["calibration"].to(device)
        gt_depth = batch["dmap"].to(device)
        gt_norm_b = batch["norm"].to(device)
        idxs = batch["idx"]

        out = model(imgs, calibs)
        pred_depth = out["depth"]
        pred_norm = out["normal"]

        all_norm_mse.append(F.mse_loss(pred_norm, gt_norm_b).item())
        all_depth_mse.append(F.mse_loss(pred_depth, gt_depth).item())

        pred_n_un = pred_norm.cpu() * torch.tensor(norm_std).view(1, 3, 1, 1) + torch.tensor(norm_mu).view(1, 3, 1, 1)
        gt_n_un = gt_norm_b.cpu() * torch.tensor(norm_std).view(1, 3, 1, 1) + torch.tensor(norm_mu).view(1, 3, 1, 1)
        ang_mean, _ = angular_error_deg(pred_n_un, gt_n_un)
        all_ang_mean.append(ang_mean)

        pred_d_un = pred_depth.cpu() * torch.tensor(dmap_std).view(1, 1, 1, 1) + torch.tensor(dmap_mu).view(1, 1, 1, 1)
        gt_d_un = gt_depth.cpu() * torch.tensor(dmap_std).view(1, 1, 1, 1) + torch.tensor(dmap_mu).view(1, 1, 1, 1)
        dm = depth_metrics(pred_d_un, gt_d_un)
        if dm:
            all_depth_mae.append(dm["MAE"])

        if vis_count < num_vis:
            for i in range(imgs.size(0)):
                gidx = idxs[i].item()
                if gidx in vis_indices:
                    obj_name = _obj_name_from_dataset(ds, gidx)
                    vis_dpt(imgs[i].cpu(),
                            pred_depth[i].cpu(), gt_depth[i].cpu(),
                            pred_norm[i].cpu(), gt_norm_b[i].cpu(),
                            save_path, vis_count, obj_name,
                            unnorm_fn=unnorm_fn)
                    vis_count += 1

    metrics = {
        "Normal MSE (normalized)": np.mean(all_norm_mse),
        "Depth MSE (normalized)": np.mean(all_depth_mse),
        "Angular Error Mean (deg)": np.mean(all_ang_mean),
        "Depth MAE (mm)": np.mean(all_depth_mae) / 255.0 if all_depth_mae else float("nan"),
    }
    return metrics


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True, choices=["sitr", "dpt", "dinov2", "dinov3", "dav2"])
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--weights", type=str, default=None,
                   help="SITR checkpoint (mode=sitr)")
    p.add_argument("--encoder-weights", type=str, default=None,
                   help="SITR encoder checkpoint (mode=dpt)")
    p.add_argument("--decoder-weights", type=str, default=None,
                   help="DPT decoder checkpoint (mode=dpt/dinov2)")
    p.add_argument("--dinov2-model", type=str, default="dinov2_vitb14",
                   choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"])
    p.add_argument("--dinov3-model", type=str, default="dinov3_vitl16")
    p.add_argument("--dinov3-weights", type=str, default=None,
                   help="Local .pth path for DINOv3 pretrained backbone weights")
    p.add_argument("--dav2-weights", type=str, default=None,
                   help="DAv2 full checkpoint .pth (encoder extracted automatically)")
    p.add_argument("--dav2-dinov2-model", type=str, default="dinov2_vitl14",
                   help="DINOv2 backbone used by DAv2 (vits14/vitb14/vitl14)")
    p.add_argument("--val-objects", type=str, nargs="+", default=None)
    p.add_argument("--gt-norm", action="store_true", default=False)
    p.add_argument("--calibration-config", type=int, default=18)
    p.add_argument("--raw-input", action="store_true", default=False)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-vis", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--save-path", type=str, default="eval_results")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)

    # ── auto-config for dinov2/dinov3 ────────────────────────────────────────
    if args.mode in ("dinov2", "dinov3", "dav2"):
        args.raw_input = True
        args.calibration_config = 0

    if args.raw_input and args.calibration_config == 18:
        args.calibration_config = 19

    # ── dataset ───────────────────────────────────────────────────────────────
    if args.mode in ("dinov2", "dinov3", "dav2"):
        img_xform = T.Compose([T.ToTensor(), T.Normalize(mean=imagenet_mu, std=imagenet_std)])
    elif args.raw_input:
        img_xform = T.Compose([T.ToTensor(), T.Normalize(mean=raw_mu, std=raw_std)])
    else:
        img_xform = None

    ds = sim_dataset_nested(
        path=args.data_path,
        augment=False,
        calibration_config=args.calibration_config,
        sendTwo=False,
        use_gt_norm=args.gt_norm,
        include_objects=args.val_objects,
        raw_input=args.raw_input,
        transforms=img_xform,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"Eval dataset: {len(ds)} samples, objects: {ds.objects}")

    # ── model ─────────────────────────────────────────────────────────────────
    if args.mode == "sitr":
        model = SITR_base(num_calibration=args.calibration_config)
        ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        model = model.to(device).eval()
        print(f"Loaded SITR from {args.weights}")

        metrics = eval_sitr(model, loader, device, args.save_path, args.num_vis)

    elif args.mode == "dpt":
        sitr = SITR_base(num_calibration=args.calibration_config)
        ckpt = torch.load(args.encoder_weights, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        sitr.load_state_dict(state)
        print(f"Loaded SITR encoder from {args.encoder_weights}")

        model = SITRWithDPT(sitr, embed_dim=768, features=256).to(device)

        dpt_ckpt = torch.load(args.decoder_weights, map_location="cpu", weights_only=False)
        dec_state = dpt_ckpt["decoder"] if "decoder" in dpt_ckpt else dpt_ckpt
        model.decoder.load_state_dict(dec_state)
        print(f"Loaded DPT decoder from {args.decoder_weights}")

        if "encoder" in dpt_ckpt:
            model.encoder.sitr.load_state_dict(dpt_ckpt["encoder"])
            print("  (encoder weights updated from DPT checkpoint — fine-tuned encoder)")

        model.eval()

        metrics = eval_dpt(model, loader, device, args.save_path, args.num_vis,
                           unnorm_fn=unnorm_raw if args.raw_input else None)

    elif args.mode == "dinov2":
        model = DINOv2WithDPT(model_name=args.dinov2_model, features=256).to(device)

        dpt_ckpt = torch.load(args.decoder_weights, map_location="cpu", weights_only=False)
        dec_state = dpt_ckpt["decoder"] if "decoder" in dpt_ckpt else dpt_ckpt
        model.decoder.load_state_dict(dec_state)
        print(f"Loaded DINOv2 ({args.dinov2_model}) + DPT decoder from {args.decoder_weights}")

        model.eval()

        metrics = eval_dpt(model, loader, device, args.save_path, args.num_vis,
                           unnorm_fn=unnorm_imagenet)

    elif args.mode == "dinov3":
        model = DINOv3WithDPT(
            model_name=args.dinov3_model,
            weights=args.dinov3_weights,
            features=256,
        ).to(device)

        dpt_ckpt = torch.load(args.decoder_weights, map_location="cpu", weights_only=False)
        dec_state = dpt_ckpt["decoder"] if "decoder" in dpt_ckpt else dpt_ckpt
        model.decoder.load_state_dict(dec_state)
        print(f"Loaded DINOv3 ({args.dinov3_model}) + DPT decoder from {args.decoder_weights}")

        model.eval()

        metrics = eval_dpt(model, loader, device, args.save_path, args.num_vis,
                           unnorm_fn=unnorm_imagenet)

    elif args.mode == "dav2":
        layer_indices_map = {
            "dinov2_vits14": (2, 5, 8, 11),
            "dinov2_vitb14": (2, 5, 8, 11),
            "dinov2_vitl14": (4, 11, 17, 23),
        }
        layer_indices = layer_indices_map.get(args.dav2_dinov2_model, (2, 5, 8, 11))
        model = DAv2WithDPT(
            model_name=args.dav2_dinov2_model,
            weights=args.dav2_weights,
            features=256,
            layer_indices=layer_indices,
        ).to(device)

        dpt_ckpt = torch.load(args.decoder_weights, map_location="cpu", weights_only=False)
        dec_state = dpt_ckpt["decoder"] if "decoder" in dpt_ckpt else dpt_ckpt
        model.decoder.load_state_dict(dec_state)
        print(f"Loaded DAv2 ({args.dav2_dinov2_model}) + DPT decoder from {args.decoder_weights}")

        model.eval()

        metrics = eval_dpt(model, loader, device, args.save_path, args.num_vis,
                           unnorm_fn=unnorm_imagenet)

    # ── print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Results ({args.mode.upper()}) on objects: {args.val_objects or 'all'}")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:35s} {v:.4f}")

    report_path = os.path.join(args.save_path, "metrics.txt")
    with open(report_path, "w") as f:
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Objects: {args.val_objects or 'all'}\n")
        f.write(f"Dataset size: {len(ds)}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"\nMetrics saved → {report_path}")
    print(f"Visualizations saved → {args.save_path}/")


if __name__ == "__main__":
    main()
