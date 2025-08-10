#!/usr/bin/env python
"""Test VAE output shape."""

import numpy as np
import jax.numpy as jnp
from diffusers import FlaxAutoencoderKL

# Load VAE
print("Loading VAE...")
vae, vae_params = FlaxAutoencoderKL.from_pretrained("pcuenq/sd-vae-ft-mse-flax", dtype=jnp.float32)

# Create test image
image = np.random.randn(1, 256, 256, 3).astype(np.float32)
print(f"Input image shape: {image.shape}")

# Convert to channels-first for VAE
image_cf = np.transpose(image, (0, 3, 1, 2))
print(f"Channels-first shape: {image_cf.shape}")

# Encode
latent = vae.apply(
    {'params': vae_params},
    jnp.array(image_cf),
    method=vae.encode
).latent_dist.mean

print(f"VAE output shape: {latent.shape}")

# Scale
latent = latent * 0.18215
print(f"Scaled shape: {latent.shape}")

# Convert back to channels-last
latent_cl = np.transpose(latent, (0, 2, 3, 1))
print(f"Channels-last shape: {latent_cl.shape}")

print("\nExpected shape for DiT: (batch, height, width, channels)")
print(f"Got: {latent_cl.shape}")
print(f"Correct? {latent_cl.shape == (1, 32, 32, 4)}")