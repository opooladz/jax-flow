#!/usr/bin/env python
"""Test training with CIFAR-10 to ensure everything compiles."""

import os
import sys

# Set minimal batch size and steps for testing
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

# Run training with minimal settings
import subprocess

cmd = [
    sys.executable,
    "train_flow.py",
    "--dataset_name", "cifar10",
    "--batch_size", "4",
    "--max_steps", "2",
    "--log_interval", "1",
    "--eval_interval", "10000",
    "--save_interval", "10000",
    "--model.use_stable_vae", "0",  # Don't use VAE for simplicity
    "--model.preset", "debug",  # Use smallest model
    "--wandb.offline", "True",  # Use offline mode for testing
]

print("Running command:")
print(" ".join(cmd))
print()

result = subprocess.run(cmd, capture_output=False, text=True)
sys.exit(result.returncode)