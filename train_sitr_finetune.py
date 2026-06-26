"""
train_sitr_finetune.py: DDP 版，支援多 GPU 訓練

Usage (2 GPU: cuda:0 and cuda:3)
-----
CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node=2 --master_port=29503 train_sitr_finetune.py --data-path /media/hdd/ihsuan/gsrl/datasets/renders --pretrain-weights /media/hdd/ihsuan/gsrl/datasets/checkpoints/SITR_B18.pth --save-path output_checkpoints/sitr_finetune --lr 1e-5 --epochs 10 --num-workers 4 --num-samples 3000 2>&1 | tee train.log

CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node=2 --master_port=29510 train_sitr_finetune.py \
    --data-path /media/hdd/ihsuan/gsrl/datasets/renders \
    --pretrain-weights /media/hdd/ihsuan/gsrl/datasets/checkpoints/SITR_B18.pth \
    --save-path output_checkpoints/sitr_finetune_v2 \
    --lr 1e-5 \
    --epochs 10 \
    --num-workers 4 \
    --num-samples 10000 \
    2>&1 | tee train_v2.log

CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node=2 --master_port=29511 train_sitr_finetune.py \
    --data-path /media/hdd/ihsuan/gsrl/datasets/renders \
    --pretrain-weights /media/hdd/ihsuan/gsrl/datasets/checkpoints/SITR_B18.pth \
    --save-path output_checkpoints/sitr_finetune_v2 \
    --resume output_checkpoints/sitr_finetune_v2/latest.pth \
    --lr 1e-5 \
    --epochs 10 \
    --num-workers 4 \
    --num-samples 10000 \
    2>&1 | tee train_resume2.log

Usage (單 GPU)
-----
python3 train_sitr_finetune.py --data-path /media/hdd/ihsuan/gsrl/datasets/renders --pretrain-weights /media/hdd/ihsuan/gsrl/datasets/checkpoints/SITR_B18.pth --save-path output_checkpoints/sitr_finetune --lr 1e-5 --epochs 10 --device cuda:0 --num-workers 4 2>&1 | tee train.log

"""

import argparse
import os
import time
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

from dataloaders import (sim_dataset, sim_dataset_nested,
                         sample_mu, sample_std, norm_mu, norm_std,
                         raw_mu, raw_std)
from models.networks import SITR_base
from models.losses import SupConLoss
import torchvision.transforms as T


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
    """Warmup LR linearly, then switch to ReduceLROnPlateau."""

    def __init__(self, optimizer, warmup_epochs, warmup_lr, base_lr,
                 factor=0.5, plateau_patience=10, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor,
            patience=plateau_patience, min_lr=min_lr, verbose=True,
        )
        self._epoch = 0
        self._set_lr(warmup_lr)

    def _set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def step(self, val_loss=None):
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            frac = self._epoch / max(1, self.warmup_epochs)
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * frac
            self._set_lr(lr)
        else:
            self.plateau.step(val_loss)

    def state_dict(self):
        return {"epoch": self._epoch, "plateau": self.plateau.state_dict()}

    def load_state_dict(self, d):
        self._epoch = d["epoch"]
        self.plateau.load_state_dict(d["plateau"])


def plot_loss_curves(history, save_path):
    epochs     = history["epochs"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("SITR Fine-tuning – Loss Curves", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], color="#2563eb", linewidth=1.8, label="Train")
    ax.plot(epochs, history["val_loss"],   color="#dc2626", linewidth=1.8, linestyle="--", label="Val")
    best_idx = history["val_loss"].index(min(history["val_loss"]))
    ax.scatter(epochs[best_idx], history["val_loss"][best_idx], color="#dc2626", s=80, zorder=5,
               label=f"Best {history['val_loss'][best_idx]:.4f}")
    ax.set_title("Total Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history["l_normal"], color="#16a34a", linewidth=1.8)
    ax.set_title("Normal Loss (MSE)"); ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history["l_scl"], color="#9333ea", linewidth=1.8)
    ax.set_title("Contrastive Loss"); ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)

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

def train_one_epoch(model, loader, optimizer, scaler, normal_criterion,
                    contrastive_criterion, lambda_normal, lambda_scl,
                    device, amp_enabled, epoch, is_ddp):
    model.train()
    if is_ddp:
        loader.sampler.set_epoch(epoch)

    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scl_meter  = AverageMeter()
    t0 = time.time()

    for step, batch in enumerate(loader):
        imgs   = batch["sample"].to(device, non_blocking=True)
        calibs = batch["calibration"].to(device, non_blocking=True)
        norms  = batch["norm"].to(device, non_blocking=True)
        labels = batch["idx"].to(device, non_blocking=True)

        img_a,   img_b   = imgs[:, 0],   imgs[:, 1]
        calib_a, calib_b = calibs[:, 0], calibs[:, 1]
        norm_a,  norm_b  = norms[:, 0],  norms[:, 1]
        B = img_a.size(0)

        optimizer.zero_grad()
        with autocast("cuda", enabled=amp_enabled):
            out_a = model(img_a, calib_a)
            out_b = model(img_b, calib_b)
            l_normal = (normal_criterion(out_a["proj"], norm_a) +
                        normal_criterion(out_b["proj"], norm_b)) * 0.5
            feats = torch.stack([out_a["cls_token"], out_b["cls_token"]], dim=1)
            l_scl = contrastive_criterion(feats, labels)
            loss = lambda_normal * l_normal + lambda_scl * l_scl

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), B)
        norm_meter.update(l_normal.item(), B)
        scl_meter.update(l_scl.item(), B)

        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [epoch {epoch:03d} | step {step:04d}/{len(loader):04d}]"
                  f"  loss={loss_meter.avg:.4f}"
                  f"  l_norm={norm_meter.avg:.4f}"
                  f"  l_scl={scl_meter.avg:.4f}"
                  f"  ({elapsed:.1f}s)")

    return loss_meter.avg, norm_meter.avg, scl_meter.avg


@torch.no_grad()
def validate(model, loader, normal_criterion, contrastive_criterion,
             lambda_normal, lambda_scl, device, amp_enabled):
    model.eval()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    for batch in loader:
        img_a   = batch["sample"][:, 0].to(device)
        img_b   = batch["sample"][:, 1].to(device)
        calib_a = batch["calibration"][:, 0].to(device)
        calib_b = batch["calibration"][:, 1].to(device)
        norm_a  = batch["norm"][:, 0].to(device)
        norm_b  = batch["norm"][:, 1].to(device)
        labels  = batch["idx"].to(device)
        with autocast("cuda", enabled=amp_enabled):
            out_a = model(img_a, calib_a)
            out_b = model(img_b, calib_b)
            l_normal = (normal_criterion(out_a["proj"], norm_a) +
                        normal_criterion(out_b["proj"], norm_b)) * 0.5
            feats = torch.stack([out_a["cls_token"], out_b["cls_token"]], dim=1)
            l_scl = contrastive_criterion(feats, labels)
            loss = lambda_normal * l_normal + lambda_scl * l_scl
        loss_meter.update(loss.item(), img_a.size(0))
        norm_meter.update(l_normal.item(), img_a.size(0))
    return loss_meter.avg, norm_meter.avg


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path",          type=str, required=True)
    p.add_argument("--layout",             type=str, default="flat", choices=["flat", "nested"],
                   help="flat: path/sensor_XXXX/...（原本）；nested: path/<obj>/session_xxx/sensor_0000/...（gs_blender 產生的）")
    p.add_argument("--gt-norm",            action="store_true", default=False,
                   help="nested 模式：用 norms/xxxx_gt.png（真實幾何法線）當目標；預設用 norms/xxxx.png（gel 渲染法線）")
    p.add_argument("--val-split",          type=float, default=0.05,
                   help="隨機切時的 val 比例（沒給 --val-objects 時用）")
    p.add_argument("--val-objects",        type=str, nargs="+", default=None,
                   help="nested 模式：指定哪些物體當 validation（held-out by object）。給了就按物體切，否則隨機切。")
    p.add_argument("--num-sensors",        type=int, default=None)
    p.add_argument("--num-samples",        type=int, default=None)
    p.add_argument("--pretrain-weights",   type=str, default=None)
    p.add_argument("--calibration-config", type=int, default=18, choices=[0, 4, 8, 9, 18, 19])
    p.add_argument("--raw-input",         action="store_true", default=False,
                   help="Use raw tactile images without background subtraction; auto-sets calib=19")
    p.add_argument("--tactile-augment",   action="store_true", default=False,
                   help="Tactile-specific augmentation (gain/bias/grad/noise/flip/rotate)")
    p.add_argument("--epochs",             type=int,   default=500,
                   help="Max epochs (early stopping may end sooner)")
    p.add_argument("--batch-size",         type=int,   default=64)
    p.add_argument("--lr",                 type=float, default=1e-5)
    p.add_argument("--min-lr",             type=float, default=1e-7)
    p.add_argument("--warmup-epochs",      type=int,   default=5)
    p.add_argument("--weight-decay",       type=float, default=0.05)
    p.add_argument("--scheduler",          type=str, default="plateau",
                   choices=["cosine", "plateau"])
    p.add_argument("--plateau-patience",   type=int, default=10,
                   help="ReduceLROnPlateau: epochs to wait before reducing LR")
    p.add_argument("--plateau-factor",     type=float, default=0.5,
                   help="ReduceLROnPlateau: LR *= factor on plateau")
    p.add_argument("--early-stop",         type=int, default=30,
                   help="Stop if val loss doesn't improve for N epochs (0=off)")
    p.add_argument("--lambda-normal",      type=float, default=1.0)
    p.add_argument("--lambda-scl",         type=float, default=1.0)
    p.add_argument("--temperature",        type=float, default=0.07)
    p.add_argument("--amp",                action="store_true", default=True)
    p.add_argument("--device",             type=str, default=None,
                   help="單 GPU 模式用，例如 cuda:0。DDP 模式不需要設。")
    p.add_argument("--gpu-ids",            type=int, nargs="+", default=None,
                   help="DDP 模式用，例如 --gpu-ids 0 3")
    p.add_argument("--num-workers",        type=int, default=4)
    p.add_argument("--no-pin-memory",      action="store_true", default=False)
    p.add_argument("--save-path",          type=str, default="checkpoints/sitr_finetune")
    p.add_argument("--save-every",         type=int, default=1)
    p.add_argument("--resume",             type=str, default=None)
    p.add_argument("--seed",               type=int, default=42)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── DDP 初始化 ────────────────────────────────────────────────────────────
    is_ddp   = "LOCAL_RANK" in os.environ
    is_main  = True  # 單 GPU 預設是 main

    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        # 把 DDP rank 對應到指定的 GPU
        if args.gpu_ids is not None:
            device_id = args.gpu_ids[local_rank]
        else:
            device_id = local_rank
        torch.cuda.set_device(device_id)
        dist.init_process_group(backend="nccl")
        device  = torch.device(f"cuda:{device_id}")
        is_main = (local_rank == 0)
    else:
        device = torch.device(args.device if args.device else "cuda:0")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ── raw-input auto-config ────────────────────────────────────────────────
    if args.raw_input and args.calibration_config == 18:
        args.calibration_config = 19

    if is_main:
        print(f"Device: {device}  DDP: {is_ddp}")
        os.makedirs(args.save_path, exist_ok=True)

    # ── datasets ──────────────────────────────────────────────────────────────
    if is_main:
        print("Loading dataset …")

    if args.raw_input:
        img_xform  = T.Compose([T.ToTensor(), T.Normalize(mean=raw_mu, std=raw_std)])
        norm_xform = T.Compose([T.ToTensor(), T.Normalize(mean=norm_mu, std=norm_std)])
    else:
        img_xform  = T.Compose([T.ToTensor(), T.Normalize(mean=sample_mu, std=sample_std)])
        norm_xform = T.Compose([T.ToTensor(), T.Normalize(mean=norm_mu, std=norm_std)])

    def build_dataset(augment, include_objects=None):
        if args.layout == "nested":
            return sim_dataset_nested(
                path=args.data_path,
                augment=augment,
                transforms=img_xform,
                norm_transforms=norm_xform,
                calibration_config=args.calibration_config,
                sendTwo=True,
                num_samples=args.num_samples,
                use_gt_norm=args.gt_norm,
                include_objects=include_objects,
                raw_input=args.raw_input,
                tactile_augment=augment and args.tactile_augment,
            )
        return sim_dataset(
            path=args.data_path,
            augment=augment,
            calibration_config=args.calibration_config,
            sendTwo=True,
            num_samples=args.num_samples,
            num_sensors=args.num_sensors,
        )

    if args.layout == "nested" and args.val_objects:
        # ── 按物體切 train/val（held-out by object，量對新物體的泛化）──
        import glob as _glob
        all_objs = sorted({os.path.basename(os.path.dirname(os.path.dirname(p)))
                           for p in _glob.glob(os.path.join(args.data_path, '*', 'session_*', 'sensor_*'))})
        val_objs = list(args.val_objects)
        missing = [o for o in val_objs if o not in all_objs]
        if missing:
            raise ValueError(f"--val-objects 不存在: {missing}\n可用物體: {all_objs}")
        train_objs = [o for o in all_objs if o not in set(val_objs)]
        if is_main:
            print(f"按物體切：train {len(train_objs)} 物體 / val {len(val_objs)} 物體（互不重疊）")
            print(f"  val objects  : {val_objs}")
            print(f"  train objects: {train_objs}")
        train_ds     = build_dataset(augment=True,  include_objects=train_objs)
        val_ds_noaug = build_dataset(augment=False, include_objects=val_objs)
    else:
        # ── 隨機切（原本行為，in-distribution val）──
        full_dataset = build_dataset(augment=True)
        n_val   = max(1, int(len(full_dataset) * args.val_split))
        n_train = len(full_dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed)
        )
        val_ds_noaug = build_dataset(augment=False)
        val_ds_noaug = torch.utils.data.Subset(val_ds_noaug, val_ds.indices)

    if is_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True,
                                           seed=args.seed, drop_last=True)
        val_sampler   = DistributedSampler(val_ds_noaug, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        val_sampler   = None
        shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=shuffle, sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(not args.no_pin_memory), drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds_noaug, batch_size=args.batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=(not args.no_pin_memory),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )

    if is_main:
        print(f"  train: {len(train_ds):,}   val: {len(val_ds_noaug):,}")

    # ── model ─────────────────────────────────────────────────────────────────
    if is_main:
        print("Building model …")
    model = SITR_base(num_calibration=args.calibration_config).to(device)
    if is_main:
        print(f"  params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    if is_ddp:
        model = DDP(model, device_ids=[device_id])

    normal_criterion      = nn.MSELoss()
    contrastive_criterion = SupConLoss(temperature=args.temperature)

    # DDP 時 lr 按 GPU 數量 scale
    effective_lr = args.lr * (dist.get_world_size() if is_ddp else 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr,
                              weight_decay=args.weight_decay,
                              betas=(0.9, 0.999))
    if args.scheduler == "cosine":
        scheduler = get_scheduler(optimizer, args.warmup_epochs, args.epochs,
                                   warmup_lr=1e-8, base_lr=effective_lr, min_lr=args.min_lr)
    else:
        scheduler = WarmupThenPlateau(
            optimizer, args.warmup_epochs,
            warmup_lr=1e-8, base_lr=effective_lr,
            factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            min_lr=args.min_lr,
        )
    scaler = GradScaler("cuda", enabled=args.amp)

    # ── weights loading ───────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        raw_model = model.module if is_ddp else model
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if is_main:
            print(f"  Resumed from epoch {start_epoch}")

    elif args.pretrain_weights is not None:
        if is_main:
            print(f"Loading pretrained weights: {args.pretrain_weights}")
        state = torch.load(args.pretrain_weights, map_location=device)
        raw_model = model.module if is_ddp else model
        model_state = raw_model.state_dict()
        compatible = {k: v for k, v in state.items()
                      if k in model_state and v.shape == model_state[k].shape}
        skipped = [k for k in state if k not in compatible]
        raw_model.load_state_dict(compatible, strict=False)
        if is_main:
            if skipped:
                print(f"  Skipped {len(skipped)} keys (shape mismatch): {skipped}")
            print(f"  Loaded {len(compatible)}/{len(state)} keys. lr={effective_lr:.1e}")

    # ── history & log（只有 rank 0 寫）────────────────────────────────────────
    history = {"epochs": [], "train_loss": [], "l_normal": [],
               "l_scl": [], "val_loss": [], "lr": []}

    if is_main:
        log_path = os.path.join(args.save_path, "train_log.csv")
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,l_normal,l_scl,val_loss,lr\n")

    # ── training loop ─────────────────────────────────────────────────────────
    epochs_no_improve = 0

    if is_main:
        n_gpus = dist.get_world_size() if is_ddp else 1
        print("\n" + "=" * 60)
        input_mode = "RAW (no bg-sub)" if args.raw_input else "DIFF (bg-sub)"
        aug_mode = "tactile" if args.tactile_augment else "standard"
        print(f"SITR Fine-tuning  |  GPUs={n_gpus}  epochs={args.epochs}"
              f"  bs={args.batch_size}×{n_gpus}={args.batch_size*n_gpus}"
              f"  lr={effective_lr:.1e}"
              f"  scheduler={args.scheduler}"
              f"  early_stop={args.early_stop}")
        print(f"Input: {input_mode}  calib={args.calibration_config}"
              f"  augment={aug_mode}")
        print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        current_lr = scheduler.get_last_lr()[0]

        if is_main:
            print(f"\nEpoch {epoch:03d}/{args.epochs-1}  lr={current_lr:.2e}")

        train_loss, l_normal, l_scl = train_one_epoch(
            model, train_loader, optimizer, scaler,
            normal_criterion, contrastive_criterion,
            args.lambda_normal, args.lambda_scl,
            device, args.amp, epoch, is_ddp)

        val_loss, val_norm = validate(
            model, val_loader,
            normal_criterion, contrastive_criterion,
            args.lambda_normal, args.lambda_scl,
            device, args.amp
        )
        if args.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # DDP：把各 GPU 的 val_loss 平均
        if is_ddp:
            val_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_tensor.item()

        if is_main:
            print(f"  → train={train_loss:.4f}  l_norm={l_normal:.4f}"
                f"  l_scl={l_scl:.4f}  val={val_loss:.4f}  val_norm={val_norm:.4f}")

            history["epochs"].append(epoch)
            history["train_loss"].append(train_loss)
            history["l_normal"].append(l_normal)
            history["l_scl"].append(l_scl)
            history["val_loss"].append(val_loss)
            history["lr"].append(current_lr)

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.6f},{l_normal:.6f},"
                        f"{l_scl:.6f},{val_loss:.6f},{current_lr:.2e}\n")

            raw_model = model.module if is_ddp else model

            def save_ckpt(path):
                torch.save({
                    "epoch":         epoch,
                    "model":         raw_model.state_dict(),
                    "optimizer":     optimizer.state_dict(),
                    "scheduler":     scheduler.state_dict(),
                    "scaler":        scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "args":          vars(args),
                }, path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
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

    if is_main:
        print(f"\nDone. Best val loss: {best_val_loss:.4f}")
        plot_loss_curves(history, args.save_path)
        raw_model = model.module if is_ddp else model
        encoder_path = os.path.join(args.save_path, "sitr_encoder.pth")
        torch.save(raw_model.state_dict(), encoder_path)
        print(f"Encoder saved → {encoder_path}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()