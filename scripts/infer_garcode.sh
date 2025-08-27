CUDA_VISIBLE_DEVICES=0 \
python infer_garcode.py --distribution_type "single" \
    --dataroot data \
    --experiment_name test_garcode \
    --model_type "DiT-tiny" \
    --npoints 8192 \
    --bs 1 \
    --lr 1e-4 \
    --vizIter 2000 \
    --workers 1 \
    --start_idx 0 \
    --model checkpoints/GARCODE/epoch_798.pth