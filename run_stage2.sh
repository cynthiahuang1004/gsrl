#!/bin/bash
# Stage 2: DPT + Pose training after SITR finetune completes
ENCODER=output_checkpoints/20260708_sitr_finetune/best.pth
PYTHON=/home/shared/miniconda3/envs/sitr/bin/python

echo "Waiting for SITR finetune to finish..."
while tmux has-session -t sitr_ft 2>/dev/null; do
    sleep 60
done
echo "SITR finetune done. Starting DPT + Pose training..."

# DPT decoder on GPU 2
mkdir -p output_checkpoints/20260708_dpt_sitr
$PYTHON -u train_dpt_sitr.py \
    --data-path /media/hdd2/ihsuan/gs_blender/renders \
    --layout nested \
    --gt-norm \
    --val-every 20 \
    --encoder sitr \
    --encoder-weights $ENCODER \
    --calibration-config 0 \
    --raw-input \
    --tactile-augment \
    --gel-spin-deg 180 \
    --center-crop \
    --depth-from-npy \
    --kendall \
    --lambda-grad 0.0 \
    --epochs 200 \
    --batch-size 64 \
    --lr 2e-4 \
    --weight-decay 0.05 \
    --scheduler plateau \
    --plateau-patience 10 \
    --early-stop 30 \
    --amp \
    --num-workers 4 \
    --device cuda:2 \
    --save-path output_checkpoints/20260708_dpt_sitr \
    --save-every 10 \
    2>&1 | tee output_checkpoints/20260708_dpt_sitr/train.log &

# Pose head on GPU 3
mkdir -p output_checkpoints/20260708_pose_sitr
$PYTHON -u train_pose_sitr.py \
    --data-path /media/hdd2/ihsuan/gs_blender/renders \
    --encoder-weights $ENCODER \
    --calibration-config 0 \
    --tactile-augment \
    --gel-spin-deg 180 \
    --center-crop \
    --pose-mode regression \
    --kendall \
    --epochs 200 \
    --batch-size 64 \
    --lr 2e-4 \
    --device cuda:3 \
    --save-path output_checkpoints/20260708_pose_sitr \
    2>&1 | tee output_checkpoints/20260708_pose_sitr/train.log &

wait
echo "All training complete."
