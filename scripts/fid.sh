!/bin/bash
export CUDA_VISIBLE_DEVICES=7

python fid.py \
  --dataset1 'PATH TO REAL TRAIN DATA' \
  --dataset2 'PATH TO SYNTHETIC TRAIN DATA' \
  --batch-size 64 \
  --workers 8 \
  --device cuda