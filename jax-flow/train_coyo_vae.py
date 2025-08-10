#!/usr/bin/env python
"""Train text-to-image on COYO-700M with SD-VAE latent space."""

import os
import sys

# Quick config for testing or full training
QUICK_TEST = "--test" in sys.argv

if QUICK_TEST:
    print("Running in TEST mode with minimal settings...")
    config = {
        "batch_size": 4,
        "image_size": 256,
        "max_steps": 10,
        "use_vae": True,
        "model_preset": "debug",
        "log_interval": 1,
    }
else:
    print("Running full training...")
    config = {
        "batch_size": 256,  # For TPU v3-8
        "image_size": 256,
        "max_steps": 500000,
        "use_vae": True,
        "model_preset": "large",
        "log_interval": 100,
    }

# Build command
cmd = [
    sys.executable,
    "train_text_to_image.py",
    "--dataset_name", "coyo",
    "--batch_size", str(config["batch_size"]),
    "--image_size", str(config["image_size"]),
    "--max_steps", str(config["max_steps"]),
    "--log_interval", str(config["log_interval"]),
    "--save_interval", "10000",
    "--model.preset", config["model_preset"],
    "--use_stable_vae", "1" if config["use_vae"] else "0",
    "--wandb.project", "coyo_vae_flow",
    "--wandb.name", f"coyo_{'test' if QUICK_TEST else 'full'}",
]

# For VAE latent space, adjust patch size
if config["use_vae"]:
    # VAE outputs 32x32x4 for 256x256 input
    # So patch_size should be smaller
    cmd.extend(["--model.patch_size", "2"])
else:
    # For pixel space 256x256x3
    cmd.extend(["--model.patch_size", "8"])

print("\nConfiguration:")
print(f"  Dataset: COYO-700M")
print(f"  Image size: {config['image_size']}x{config['image_size']}")
print(f"  Using VAE: {config['use_vae']}")
print(f"  Batch size: {config['batch_size']}")
print(f"  Model preset: {config['model_preset']}")
print(f"  Max steps: {config['max_steps']}")

print("\nCommand:")
print(" ".join(cmd))

if not QUICK_TEST:
    print("\n⚠️  This will download COYO-700M and train for real!")
    print("Add --test flag for quick testing")
    response = input("\nContinue? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)

# Run training
import subprocess
result = subprocess.run(cmd, capture_output=False, text=True)
sys.exit(result.returncode)