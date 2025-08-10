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
    "--batch_size", "4",
    "--image_size", "64",
    "--max_steps", "5",
    "--log_interval", "1",
    "--save_interval", "10000",
    "--model.preset", "debug",
    "--use_stable_vae", "1",  # Test VAE path
    "--model.patch_size", "2",  # Small patches for VAE latents
    "--wandb.offline", "True",
]

print("Testing COYO pipeline with mock data...")
print("Command:")
print(" ".join(cmd))
print()

result = subprocess.run(cmd, capture_output=False, text=True)
sys.exit(result.returncode)