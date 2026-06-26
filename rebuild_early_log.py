"""Reconstruct epoch 0-11 metrics for sitr_finetune_raw by evaluating saved checkpoints."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from dataloaders import sim_dataset_nested, raw_mu, raw_std
from models.networks import SITR_base
from models.losses import SupConLoss
import torchvision.transforms as T
import csv, os

CKPT_DIR = "output_checkpoints/sitr_finetune_raw"
DATA_PATH = "/media/hdd/ihsuan/gs_blender/renders"
VAL_OBJECTS = ["edge", "hex_key", "pattern_31_rod"]
DEVICE = "cuda:0"
BATCH_SIZE = 64
NUM_WORKERS = 4

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = self.count = 0
    def update(self, val, n=1):
        self.sum += val * n; self.count += n
    @property
    def avg(self):
        return self.sum / self.count if self.count else 0

@torch.no_grad()
def evaluate(model, loader, normal_criterion, contrastive_criterion, device):
    model.eval()
    loss_m, norm_m, scl_m = AverageMeter(), AverageMeter(), AverageMeter()
    for batch in tqdm(loader, leave=False, ncols=80):
        img_a   = batch["sample"][:, 0].to(device)
        img_b   = batch["sample"][:, 1].to(device)
        calib_a = batch["calibration"][:, 0].to(device)
        calib_b = batch["calibration"][:, 1].to(device)
        norm_a  = batch["norm"][:, 0].to(device)
        norm_b  = batch["norm"][:, 1].to(device)
        labels  = batch["idx"].to(device)
        with autocast("cuda"):
            out_a = model(img_a, calib_a)
            out_b = model(img_b, calib_b)
            l_normal = (normal_criterion(out_a["proj"], norm_a) +
                        normal_criterion(out_b["proj"], norm_b)) * 0.5
            feats = torch.stack([out_a["cls_token"], out_b["cls_token"]], dim=1)
            l_scl = contrastive_criterion(feats, labels)
            loss = l_normal + l_scl
        B = img_a.size(0)
        loss_m.update(loss.item(), B)
        norm_m.update(l_normal.item(), B)
        scl_m.update(l_scl.item(), B)
    return loss_m.avg, norm_m.avg, scl_m.avg

def main():
    tf = T.Compose([T.ToTensor(), T.Normalize(mean=raw_mu, std=raw_std)])

    all_objs = sorted(os.listdir(DATA_PATH))
    all_objs = [o for o in all_objs if os.path.isdir(os.path.join(DATA_PATH, o))]
    train_objs = [o for o in all_objs if o not in set(VAL_OBJECTS)]

    print(f"Building datasets: train={len(train_objs)} objs, val={len(VAL_OBJECTS)} objs")

    common = dict(path=DATA_PATH, augment=False, sendTwo=True,
                  transforms=tf, raw_input=True, calibration_config=19,
                  use_gt_norm=True)
    train_ds = sim_dataset_nested(**common, include_objects=train_objs)
    val_ds   = sim_dataset_nested(**common, include_objects=VAL_OBJECTS)
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False)

    normal_crit = nn.MSELoss()
    scl_crit = SupConLoss(temperature=0.07)

    model = SITR_base(num_calibration=19).to(DEVICE)

    results = []
    for ep in range(9, 12):
        ckpt_path = os.path.join(CKPT_DIR, f"epoch_{ep:04d}.pth")
        print(f"\n=== Epoch {ep} ===")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])

        print("  evaluating train...")
        tr_loss, tr_norm, tr_scl = evaluate(model, train_loader, normal_crit, scl_crit, DEVICE)
        print("  evaluating val...")
        va_loss, va_norm, va_scl = evaluate(model, val_loader, normal_crit, scl_crit, DEVICE)

        lr = 2e-05 if ep >= 5 else (ep + 1) / 5 * 2e-05
        results.append((ep, tr_loss, tr_norm, tr_scl, va_loss, lr))
        print(f"  train={tr_loss:.6f} l_norm={tr_norm:.6f} l_scl={tr_scl:.6f}  val={va_loss:.6f}")

    print("\n\n=== CSV rows (epoch 0-11) ===")
    print("epoch,train_loss,l_normal,l_scl,val_loss,lr")
    for ep, tl, tn, ts, vl, lr in results:
        print(f"{ep},{tl:.6f},{tn:.6f},{ts:.6f},{vl:.6f},{lr:.2e}")

if __name__ == "__main__":
    main()
