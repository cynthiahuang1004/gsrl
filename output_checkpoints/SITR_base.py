mkdir -p /media/hdd/ihsuan/gsrl/output_checkpoints/downstream_cls_16_2/SITR_base

python3 -c "
import torch, os

for sensor in range(7):
    ckpt_path = f'/media/hdd/ihsuan/gsrl/output_checkpoints/downstream_cls_16_2/sensor_000{sensor}/best.pth'
    out_path  = f'/media/hdd/ihsuan/gsrl/output_checkpoints/downstream_cls_16_2/SITR_base/sensor_000{sensor}.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    torch.save(ckpt['model'], out_path)
    print(f'sensor {sensor} done')
"