#!/bin/bash

# SCENES=(bicycle bonsai counter garden kitchen room stump train truck)
# SCENES=(room)
SCENES=(flowers_)
VQ_RATIO=0.6
CODEBOOK_SIZE=8192

for SCENE in "${SCENES[@]}"   # Add more scenes as needed
do
    IMP_PATH=/u/xcui1/code/LightGaussian/output/${SCENE}
    INPUT_PLY_PATH=/u/xcui1/code/LightGaussian/output/${SCENE}/point_cloud/iteration_40000/point_cloud.ply
    SAVE_PATH=./vectree/output/${SCENE}

    CMD="CUDA_VISIBLE_DEVICES=0 python vectree/vectree.py \
    --important_score_npz_path ${IMP_PATH} \
    --input_path ${INPUT_PLY_PATH} \
    --save_path ${SAVE_PATH} \
    --vq_ratio ${VQ_RATIO} \
    --codebook_size ${CODEBOOK_SIZE} \
    "
    eval $CMD
done