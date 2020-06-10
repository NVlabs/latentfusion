#!/usr/bin/env bash

set -v

TAG=$2-$(hostname)

SAVE_DIR=$SAVE_DIR/$TAG
#SHAPENET_PATH="path-to-shapenet"
#TEXTURES_PATH="path-to-coco"


python tools/train/train_reconstruct.py \
    --histogram-interval -1 \
    --grad-histogram-interval -1 \
    --plot-interval 500 \
    --show-interval 1000 \
    --dataset-type shapenet \
    --dataset-path $SHAPENET_PATH \
    --textures-path $TEXTURES_PATH \
    --color-background-path $TEXTURES_PATH \
    --color-noise-level 0.05 \
    --depth-noise-level 0.00 \
    --gpu-id 0 \
    --dataset-gpu-id 0 \
    --num-workers 5 \
    --save-dir $SAVE_DIR \
    --base-name shapenet,no_mask_morph,fixed_eqlr,256 \
    --input-size 256 \
    --batch-size 8 \
    --batch-groups 2 \
    --batches-per-epoch 4000 \
    --num-input-views 8 \
    --num-output-views 24 \
    --optimizer adam \
    --generator-lr 0.00075 \
    --discriminator-lr 0.00075 \
    --sculptor-image-config 64,D,128,D,196,D,256,D,512,D,512,D,512:512,U,512,U,256 \
    --sculptor-camera-config 64,128,256 \
    --sculptor-object-config 256,256 \
    --sculptor-projection-type factor \
    --photographer-object-config none \
    --photographer-camera-config 256,256 \
    --photographer-image-config 256,D,512,D,512:512,U,512,U,512,U,256,U,196,U,128,U,64 \
    --photographer-projection-type factor \
    --discriminator-config 64,128,256 \
    --fuser-type gru \
    --g-depth-recon-loss-type hard_smooth_l1 \
    --g-depth-recon-loss-weight 25.0 \
    --g-depth-recon-loss-k 16384 \
    --g-depth-recon-loss-k-milestones 15,30,45,60 \
    --g-mask-recon-loss-type binary_cross_entropy \
    --g-mask-recon-loss-weight 25.0 \
    --g-mask-beta-loss-weight 0.0 \
    --g-mask-beta-loss-param 0.01 \
    --random-orientation \
    --crop-predicted-mask \
    --discriminator-input-depth \
    --generator-input-mask \
    --no-discriminator \
    --color-random-background \
    --crop-random-background \
    --mask-noise-p 0.25 \
    --predict-depth \
    --predict-mask \
    --scale-mode nearest \
    --data-parallel
