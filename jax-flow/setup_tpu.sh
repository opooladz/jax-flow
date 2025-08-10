#!/bin/bash
# Setup script for TPU environment

echo "Setting up TPU environment..."

# Install dependencies
pip install --upgrade pip
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax optax einops ml_collections wandb matplotlib tqdm
pip install datasets pillow numpy
pip install diffusers transformers accelerate
pip install jaxtyping typeguard

# Set TPU environment variables (adjust if needed)
export XLA_FLAGS="--xla_force_host_platform_device_count=8"

echo "TPU setup complete!"
echo ""
echo "To verify TPU is available, run:"
echo "python -c 'import jax; print(jax.devices())'"
echo ""
echo "You should see 8 TPU devices listed."
echo ""
echo "Quick test commands:"
echo "1. Test CIFAR-10 (fast):"
echo "   python train_flow.py --dataset_name cifar10 --batch_size 256 --max_steps 100 --model.preset debug"
echo ""
echo "2. Full training:"
echo "   ./train_tpu.sh"