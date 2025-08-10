#!/bin/bash
# Training script for TPU v3-8 with HuggingFace streaming

echo "Starting training on TPU v3-8..."

# For TPU, you'll want larger batch sizes to utilize all cores
# v3-8 has 8 cores, so batch_size should be divisible by 8

# Option 1: Train on CIFAR-10 (quick test)
python train_flow.py \
    --dataset_name cifar10 \
    --batch_size 256 \
    --max_steps 10000 \
    --log_interval 100 \
    --eval_interval 1000 \
    --save_interval 5000 \
    --model.use_stable_vae 0 \
    --model.preset big \
    --model.patch_size 8 \
    --wandb.project flow_tpu \
    --wandb.name "cifar10_tpu_test"

# Option 2: Train on TinyImageNet (longer training)
# Uncomment to use:
# python train_flow.py \
#     --dataset_name tiny-imagenet \
#     --batch_size 128 \
#     --max_steps 100000 \
#     --log_interval 100 \
#     --eval_interval 5000 \
#     --save_interval 10000 \
#     --model.use_stable_vae 0 \
#     --model.preset big \
#     --model.patch_size 8 \
#     --wandb.project flow_tpu \
#     --wandb.name "tiny_imagenet_tpu"

# Option 3: Larger model on TinyImageNet
# Uncomment to use:
# python train_flow.py \
#     --dataset_name tiny-imagenet \
#     --batch_size 64 \
#     --max_steps 200000 \
#     --log_interval 100 \
#     --eval_interval 5000 \
#     --save_interval 10000 \
#     --model.use_stable_vae 0 \
#     --model.preset semilarge \
#     --model.patch_size 8 \
#     --wandb.project flow_tpu \
#     --wandb.name "tiny_imagenet_tpu_large"