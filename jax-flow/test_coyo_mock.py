#!/usr/bin/env python
"""Test COYO training with mock data (no downloads)."""

import os
import sys

# Force mock dataset for testing
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

# Run training with mock data
import subprocess

cmd = [
    sys.executable,
    "train_text_to_image.py",
    "--dataset_name", "mock",  # Use mock dataset
    "--batch_size", "8",  # Must be divisible by number of TPU cores (8)
    "--image_size", "64",
    "--max_steps", "5",
    "--log_interval", "1",
    "--save_interval", "10000",
    "--model.preset", "debug",
    "--use_stable_vae", "0",  # Start without VAE for simplicity
    "--model.patch_size", "8",  # Normal patches for pixel space
    "--wandb.offline", "True",
]

print("Testing COYO pipeline with mock data...")
print("Command:")
print(" ".join(cmd))
print()

result = subprocess.run(cmd, capture_output=False, text=True)
sys.exit(result.returncode)