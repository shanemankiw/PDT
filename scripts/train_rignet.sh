CUDA_VISIBLE_DEVICES=0,1 \
python train_rignet.py --distribution_type "single" \
    --dataroot data \
    --experiment_name train_rignet \
    --model_type "DiT-tiny" \
    --npoints 2048 \
    --bs 64 \
    --lr 1e-4 \
    --vizIter 92 \
    --saveIter 52