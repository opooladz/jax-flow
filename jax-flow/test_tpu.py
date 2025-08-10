#!/usr/bin/env python
"""Test script specifically for TPU v3-8."""

import os
import sys
import jax

print("=" * 60)
print("TPU Test Script")
print("=" * 60)

# Check TPU devices
devices = jax.devices()
print(f"\nDetected {len(devices)} devices:")
for i, d in enumerate(devices):
    print(f"  Device {i}: {d}")

num_devices = len(devices)
print(f"\nNumber of TPU cores: {num_devices}")

# Determine batch size
if num_devices == 8:
    # TPU v3-8
    batch_sizes = {
        "small": 8,    # 1 per device
        "medium": 32,  # 4 per device  
        "large": 128,  # 16 per device
    }
    print("\nRecommended batch sizes for TPU v3-8:")
    for name, size in batch_sizes.items():
        print(f"  {name}: {size} (per-device: {size//num_devices})")
else:
    print(f"\nWarning: Expected 8 devices but found {num_devices}")
    batch_sizes = {"default": num_devices}

# Get test mode
test_mode = sys.argv[1] if len(sys.argv) > 1 else "small"
batch_size = batch_sizes.get(test_mode, batch_sizes["small"])

print(f"\nRunning test with batch_size={batch_size}")

# Build command
cmd = [
    sys.executable,
    "train_text_to_image.py",
    "--dataset_name", "mock",
    "--batch_size", str(batch_size),
    "--image_size", "64",
    "--max_steps", "10",
    "--log_interval", "1",
    "--save_interval", "10000",
    "--model.preset", "debug",
    "--use_stable_vae", "0",
    "--model.patch_size", "8",
    "--wandb.offline", "True",
]

print("\nCommand:")
print(" ".join(cmd))
print("\nStarting training...")
print("-" * 60)

import subprocess
result = subprocess.run(cmd, capture_output=False, text=True)

if result.returncode == 0:
    print("-" * 60)
    print("✅ Test successful!")
    print("\nNext steps:")
    print("1. Try larger batch size: python test_tpu.py medium")
    print("2. Test with VAE: python train_coyo_vae.py --test")
    print("3. Full training: python train_coyo_vae.py")
else:
    print("-" * 60)
    print("❌ Test failed!")
    print("Check the error messages above.")

sys.exit(result.returncode)