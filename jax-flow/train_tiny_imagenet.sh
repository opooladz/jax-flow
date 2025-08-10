#!/bin/bash
# Training script for TinyImageNet using HuggingFace streaming

echo "Starting TinyImageNet training with HuggingFace streaming..."

# Activate the virtual environment
source ../.venv/bin/activate

# Run training with TinyImageNet
python train_flow.py \
    --dataset_name tiny-imagenet \
    --batch_size 32 \
    --max_steps 100000 \
    --log_interval 100 \
    --eval_interval 5000 \
    --save_interval 10000 \
    --model.use_stable_vae 0 \
    --model.preset big \
    --model.patch_size 8 \
    --wandb.project flow_tiny_imagenet \
    --wandb.name "tiny_imagenet_flow"

echo "Training completed!"