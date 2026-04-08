"""
train_downstream.py – Fine-tuning script for SITR downstream tasks

Paper : "Sensor-Invariant Tactile Representation", Gupta et al., ICLR 2025
Repo  : https://github.com/hgupt3/gsrl

Supports two tasks:
  1. classification  – 16-class object classification (Top-1 accuracy)
  2. pose_estimation – 3-DoF contact pose (RMSE in mm)

The SITR encoder is kept frozen; only the task-specific decoder head is trained.

Usage (classification)
----------------------
python train_downstream.py \
    --task classification \
    --encoder-weights /media/hdd/ihsuan/gsrl/output_checkpoints/sitr_finetune/sitr_encoder.pth \
    --data-path /media/hdd/ihsuan/gsrl/datasets/classification_dataset/train_set \
    --val-path  /media/hdd/ihsuan/gsrl/datasets/classification_dataset/val_set \
    --save-path /media/hdd/ihsuan/gsrl/output_checkpoints/downstream_cls \
    --train-sensor 0 \
    --num-classes 20 \
    --epochs 1 \
    --batch-size 64 \
    --device cuda:2

for sensor in 0 1 2 3 4 5 6; do
    python train_downstream.py \
        --task classification \
        --encoder-weights /media/hdd/ihsuan/gsrl/output_checkpoints/sitr_finetune/sitr_encoder.pth \
        --data-path /media/hdd/ihsuan/gsrl/datasets/classification_dataset/train_set \
        --val-path  /media/hdd/ihsuan/gsrl/datasets/classification_dataset/val_set \
        --save-path /media/hdd/ihsuan/gsrl/output_checkpoints/downstream_cls \
        --train-sensor $sensor \
        --num-classes 20 \
        --epochs 50 \
        --batch-size 64 \
        --device cuda:0 \
        2>&1 | tee /media/hdd/ihsuan/gsrl/output_checkpoints/downstream_cls/train_sensor${sensor}.log
done

Usage (pose estimation)
-----------------------
python train_downstream.py \
    --task pose_estimation \
    --encoder-weights /media/hdd/ihsuan/gsrl/output_checkpoints/sitr_finetune/sitr_encoder.pth \
    --data-path /media/hdd/ihsuan/gsrl/datasets/pose_dataset/train_set \
    --val-path  /media/hdd/ihsuan/gsrl/datasets/pose_dataset/val_set \
    --save-path /media/hdd/ihsuan/gsrl/output_checkpoints/pose_estimation \
    --train-sensor 0 \
    --epochs 1 \
    --batch-size 32 \
    --device cuda:2
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# ── project imports ───────────────────────────────────────────────────────────
from dataloaders import classification_dataset, pose_dataset
from models.networks import (
    SITR_base,
    classification_net,
    pose_estimation_net,
)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
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


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item() * 100.0


# ──────────────────────────────────────────────────────────────────────────────
# Classification training
# ──────────────────────────────────────────────────────────────────────────────

def train_classification(model, loader, optimizer, scaler, criterion,
                         device, amp_enabled, epoch):
    model.train()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()

    for step, batch in enumerate(loader):
        imgs   = batch['sample'].to(device)
        calibs = batch['calibration'].to(device)
        labels = batch['label'].to(device).argmax(dim=1)

        optimizer.zero_grad()
        with autocast(enabled=amp_enabled):
            logits = model(imgs, calibs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_m.update(loss.item(),               imgs.size(0))
        acc_m.update(top1_accuracy(logits.detach(), labels), imgs.size(0))

        if step % 50 == 0:
            print(f"  [epoch {epoch:03d} | {step:04d}/{len(loader):04d}]"
                  f"  loss={loss_m.avg:.4f}  acc={acc_m.avg:.2f}%")

    return loss_m.avg, acc_m.avg


@torch.no_grad()
def validate_classification(model, loader, criterion, device, amp_enabled):
    model.eval()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()

    for batch in loader:
        imgs   = batch['sample'].to(device)
        calibs = batch['calibration'].to(device)
        labels = batch['label'].to(device).argmax(dim=1)
        with autocast(enabled=amp_enabled):
            logits = model(imgs, calibs)
            loss   = criterion(logits, labels)
        loss_m.update(loss.item(),                imgs.size(0))
        acc_m.update(top1_accuracy(logits, labels), imgs.size(0))

    return loss_m.avg, acc_m.avg


# ──────────────────────────────────────────────────────────────────────────────
# Pose estimation training
# ──────────────────────────────────────────────────────────────────────────────

def train_pose(model, loader, optimizer, scaler, criterion,
               device, amp_enabled, epoch):
    """
    pose_dataset yields pairs of (initial, final) tactile observations.
    Each sample: (img1, img2, calib, delta_pose)
    where delta_pose is a (3,) tensor of (x, y, z) displacement in mm.
    """
    model.train()
    loss_m = AverageMeter()

    for step, batch in enumerate(loader):
        img1  = batch['sample_init'].to(device)
        img2  = batch['sample_final'].to(device)
        calib = batch['calibration'].to(device)
        pose  = batch['label'].to(device).float()

        optimizer.zero_grad()
        with autocast(enabled=amp_enabled):
            pred = model(img1, img2, calib)   # (B, 3)
            loss = criterion(pred, pose)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # RMSE per step
        with torch.no_grad():
            rmse = torch.sqrt(((pred.detach() - pose) ** 2).mean()).item()
        loss_m.update(rmse, img1.size(0))

        if step % 50 == 0:
            print(f"  [epoch {epoch:03d} | {step:04d}/{len(loader):04d}]"
                  f"  RMSE={loss_m.avg:.4f}mm")

    return loss_m.avg


@torch.no_grad()
def validate_pose(model, loader, device, amp_enabled):
    model.eval()
    sq_err_sum = 0.0
    count      = 0

    for batch in loader:
        img1  = batch['sample_init'].to(device)
        img2  = batch['sample_final'].to(device)
        calib = batch['calibration'].to(device)
        pose  = batch['label'].to(device).float()
        with autocast(enabled=amp_enabled):
            pred = model(img1, img2, calib)
        sq_err_sum += ((pred - pose) ** 2).sum().item()
        count      += pose.numel()          # B × 3

    return (sq_err_sum / max(count, 1)) ** 0.5   # overall RMSE in mm


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SITR Downstream Fine-tuning (classification / pose_estimation)")

    p.add_argument("--task", type=str, required=True,
                   choices=["classification", "pose_estimation"])

    # ── encoder ───────────────────────────────────────────────────────────────
    p.add_argument("--encoder-weights",    type=str, default=None,
                   help="Path to pre-trained SITR encoder weights (sitr_encoder.pth)."
                        " If omitted, encoder is randomly initialised.")
    p.add_argument("--calibration-config", type=int, default=18,
                   choices=[0, 4, 8, 9, 18])
    p.add_argument("--freeze-encoder",     action="store_true", default=True,
                   help="Keep SITR encoder frozen (paper default)")

    # ── data ──────────────────────────────────────────────────────────────────
    p.add_argument("--data-path",    type=str, required=True,
                   help="Path to training split")
    p.add_argument("--val-path",     type=str, required=True,
                   help="Path to validation split")
    p.add_argument("--train-sensor", type=int, default=0,
                   help="Sensor index to train on (decoder trained on single sensor)")

    # classification-specific
    p.add_argument("--num-classes",  type=int, default=16)
    p.add_argument("--class-list",   type=int, nargs="*", default=None,
                   help="Subset of class IDs to use (default: all)")

    # pose-specific
    p.add_argument("--random-final", action="store_true", default=False,
                   help="Use random final states in pose pairs")

    # ── training ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--amp",          action="store_true", default=True)
    p.add_argument("--device",       type=str,   default="cuda:0")
    p.add_argument("--num-workers",  type=int,   default=8)
    p.add_argument("--save-path",    type=str,   default="checkpoints/downstream")
    p.add_argument("--save-every",   type=int,   default=10)
    p.add_argument("--resume",       type=str,   default=None)
    p.add_argument("--seed",         type=int,   default=42)

    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Task : {args.task}   Device: {device}")

    save_path = os.path.join(args.save_path, f"sensor_{args.train_sensor:04d}")
    os.makedirs(save_path, exist_ok=True)

    # ── encoder ───────────────────────────────────────────────────────────────
    print("Building SITR encoder …")
    encoder = SITR_base(num_calibration=args.calibration_config)

    if args.encoder_weights is not None:
        state = torch.load(args.encoder_weights, map_location="cpu")
        encoder.load_state_dict(state)
        print(f"  Loaded encoder weights from {args.encoder_weights}")
    else:
        print("  WARNING: No encoder weights provided – using random init.")

    if args.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad_(False)
        print("  Encoder frozen.")

    # ── build task model ──────────────────────────────────────────────────────
    if args.task == "classification":
        model = classification_net(encoder, num_classes=args.num_classes).to(device)

        train_ds = classification_dataset(
            path=args.data_path,
            sensor_list=[args.train_sensor],
            class_list=args.class_list,
            calibration_config=args.calibration_config,
            augment=True,
        )
        val_ds = classification_dataset(
            path=args.val_path,
            sensor_list=[0, 1, 2, 3, 4, 5, 6],            # evaluate on all sensors
            class_list=args.class_list,
            calibration_config=args.calibration_config,
            augment=False,
        )
        criterion = nn.CrossEntropyLoss()

    elif args.task == "pose_estimation":
        model = pose_estimation_net(encoder).to(device)

        train_ds = pose_dataset(
            path=args.data_path,
            sensor_list=[args.train_sensor],
            random_final=args.random_final,
            augment=True,
        )
        val_ds = pose_dataset(
            path=args.val_path,
            sensor_list=[0, 1, 2, 3, 4, 5, 6],
            random_final=False,
            augment=False,
        )
        criterion = nn.MSELoss()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable / 1e6:.2f}M")

    # ── dataloaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    print(f"  train: {len(train_ds):,}   val: {len(val_ds):,}")

    # ── optimiser (only decoder parameters are updated) ───────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler(enabled=args.amp)

    # ── optional resume ───────────────────────────────────────────────────────
    start_epoch = 0
    best_metric = float("inf") if args.task == "pose_estimation" else -float("inf")

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt.get("best_metric", best_metric)
        print(f"  Resumed from epoch {start_epoch}")

    # ── log file ──────────────────────────────────────────────────────────────
    log_path = os.path.join(save_path, "train_log.csv")
    with open(log_path, "w") as f:
        if args.task == "classification":
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")
        else:
            f.write("epoch,train_rmse,val_rmse,lr\n")

    # ── training loop ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Fine-tuning [{args.task}] on sensor {args.train_sensor}")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch:03d}/{args.epochs - 1}  lr={lr:.2e}")

        # ── train one epoch ───────────────────────────────────────────────────
        if args.task == "classification":
            train_loss, train_acc = train_classification(
                model, train_loader, optimizer, scaler,
                criterion, device, args.amp, epoch)
            val_loss, val_acc = validate_classification(
                model, val_loader, criterion, device, args.amp)

            print(f"  → train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%"
                  f"  val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%")

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.6f},{train_acc:.4f},"
                        f"{val_loss:.6f},{val_acc:.4f},{lr:.2e}\n")

            is_best = val_acc > best_metric
            if is_best:
                best_metric = val_acc
                print(f"  ✓ new best val_acc={best_metric:.2f}%")

        else:  # pose_estimation
            train_rmse = train_pose(
                model, train_loader, optimizer, scaler,
                criterion, device, args.amp, epoch)
            val_rmse   = validate_pose(model, val_loader, device, args.amp)

            print(f"  → train_RMSE={train_rmse:.4f}mm  val_RMSE={val_rmse:.4f}mm")

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_rmse:.6f},{val_rmse:.6f},{lr:.2e}\n")

            is_best = val_rmse < best_metric
            if is_best:
                best_metric = val_rmse
                print(f"  ✓ new best val_RMSE={best_metric:.4f}mm")

        scheduler.step()

        # ── checkpointing ─────────────────────────────────────────────────────
        def save_ckpt(path):
            torch.save({
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "best_metric": best_metric,
                "args":        vars(args),
            }, path)

        if is_best:
            save_ckpt(os.path.join(save_path, "best.pth"))

        if (epoch + 1) % args.save_every == 0:
            save_ckpt(os.path.join(save_path, f"epoch_{epoch:04d}.pth"))

        save_ckpt(os.path.join(save_path, "latest.pth"))

    print("\nFine-tuning complete.")
    if args.task == "classification":
        print(f"Best val_acc : {best_metric:.2f}%")
    else:
        print(f"Best val_RMSE: {best_metric:.4f}mm")


if __name__ == "__main__":
    main()


'''
for sensor in 6; do
    python train_downstream.py \
        --task classification \
        --encoder-weights /media/hdd/ihsuan/gsrl/output_checkpoints/sitr_finetune/sitr_encoder.pth \
        --data-path /media/hdd/ihsuan/gsrl/datasets/classification_dataset/train_set \
        --val-path  /media/hdd/ihsuan/gsrl/datasets/classification_dataset/val_set \
        --save-path /media/hdd/ihsuan/gsrl/output_checkpoints/downstream_cls_16 \
        --train-sensor $sensor \
        --num-classes 16 \
        --class-list 0 2 3 4 5 7 8 9 10 11 13 14 15 16 17 18 \
        --epochs 40 \
        --batch-size 64 \
        --device cuda:2 \
        2>&1 | tee /media/hdd/ihsuan/gsrl/output_checkpoints/downstream_cls_16/train_sensor${sensor}.log
done
'''