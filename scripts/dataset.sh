#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

echo "MGD-imagenet100-IPC10"
python kl.py \
  --dataset1 'PATH TO SYNTHETIC DATA' \
  --dataset2 'PATH TO REAL DATA' \
  --batch-size 64 \
  --workers 8 \
  --device cuda
