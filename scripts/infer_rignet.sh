#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python infer_rignet_dps_x0.py --distribution_type "single" \
    --dataroot /home \
    --experiment_name rignet_test \
    --model_type "DiT-tiny" \
    --npoints 2048 \
    --manualSeed 4780 \
    --bs 1 \
    --lr 1e-4 \
    --vizIter 1 \
    --workers 1 \
    --start_idx 0 \
    --use_wandb \
    --model checkpoints/RIGNET_X0/epoch_2490.pth
