#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

python representativeness_diversity.py \
  --real_dir 'PATH TO REAL TRAIN DATA' \
  --syn_dir 'PATH TO SYNTHETIC TRAIN DATA' \