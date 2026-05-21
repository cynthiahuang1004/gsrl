import pandas as pd
import matplotlib.pyplot as plt

# 你的 CSV 路徑
csv_path = "/media/hdd/ihsuan/gsrl/output_checkpoints/pose_estimation/sensor_0003/train_log.csv"

# 讀資料
df = pd.read_csv(csv_path)

# 取出欄位
epoch = df["epoch"]
train_rmse = df["train_rmse"]
val_rmse = df["val_rmse"]

# 畫圖
plt.figure()
plt.plot(epoch, train_rmse, label="Train RMSE")
plt.plot(epoch, val_rmse, label="Validation RMSE")

# 標籤與標題
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Training vs Validation RMSE")
plt.legend()

# 存圖（可選）
plt.savefig("loss_curve_3.png")

# 顯示
plt.show()