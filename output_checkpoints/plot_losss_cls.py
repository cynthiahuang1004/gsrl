import pandas as pd
import matplotlib.pyplot as plt
import os

# ====== 路徑 ======
csv_path = "/media/hdd/ihsuan/gsrl/output_checkpoints/downstream_cls_16_2/sensor_0006/train_log.csv"

# 抓 sensor 名稱
sensor_name = os.path.basename(os.path.dirname(csv_path))

# 輸出檔名
save_path = f"downstream_cls_loss_{sensor_name}.png"

# ====== 讀資料 ======
df = pd.read_csv(csv_path)

# ====== 畫圖 ======
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- 左邊：Loss ---
axes[0].plot(df["epoch"], df["train_loss"], label="train_loss")
axes[0].plot(df["epoch"], df["val_loss"], label="val_loss")
axes[0].set_title("Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# --- 右邊：Accuracy ---
axes[1].plot(df["epoch"], df["train_acc"], label="train_acc")
axes[1].plot(df["epoch"], df["val_acc"], label="val_acc")
axes[1].set_title("Accuracy Curve")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

plt.suptitle(f"Downstream Training Results ({sensor_name})")

# 排版避免重疊
plt.tight_layout()

# ====== 存圖 ======
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Saved to {save_path}")