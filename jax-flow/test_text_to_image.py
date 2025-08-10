#!/usr/bin/env python
"""Test text-to-image training with minimal settings."""

import os
import sys

# Set minimal settings for testing
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

# Run training with minimal settings
import subprocess

cmd = [
    sys.executable,
    "train_text_to_image.py",
    "--dataset_name", "conceptual_captions",
    "--batch_size", "2",
    "--max_steps", "2",
    "--log_interval", "1",
    "--eval_interval", "10000",
    "--save_interval", "10000",
    "--image_size", "64",  # Small images for testing
    "--model.preset", "debug",  # Smallest model
    "--wandb.offline", "True",
]

print("Testing text-to-image training...")
print("Command:")
print(" ".join(cmd))
print()

result = subprocess.run(cmd, capture_output=False, text=True)
sys.exit(result.returncode)