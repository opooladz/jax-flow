#!/usr/bin/env python
"""Quick test of text-to-image pipeline with mock data."""

import sys
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

import jax
import jax.numpy as jnp
import numpy as np
from utils.text_image_datasets_mock import create_mock_dataloader
from diffusion_transformer_text import DiTText

print("Testing text-to-image pipeline with mock data...")
print("=" * 50)

# Create mock dataset
print("\n1. Creating mock dataset...")
loader = create_mock_dataloader(
    batch_size=2,
    image_size=64,
    text_embed_dim=384
)

# Get a batch
print("\n2. Getting a batch...")
batch_iter = loader.create_infinite_iterator()
batch = next(batch_iter)

print(f"   Image shape: {batch['image'].shape}")
print(f"   Text embedding shape: {batch['text_embedding'].shape}")
print(f"   Sample texts: {batch['text'][:2]}")

# Initialize model
print("\n3. Initializing DiT-Text model...")
model = DiTText(
    patch_size=16,  # Large patches for 64x64 images
    hidden_size=128,  # Small model
    depth=2,
    num_heads=4,
    mlp_ratio=2.0,
    text_dropout_prob=0.1,
    text_embed_dim=384
)

# Initialize with example
rng = jax.random.PRNGKey(42)
example_img = batch['image'][:1]
example_text = batch['text_embedding'][:1]
example_t = jnp.array([0.5])

print(f"   Input image shape: {example_img.shape}")
print(f"   Input text shape: {example_text.shape}")

params = model.init(
    {'params': rng, 'text_dropout': rng},
    example_img,
    example_t,
    example_text
)['params']

print(f"   Model parameters: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,}")

# Test forward pass
print("\n4. Testing forward pass...")
output = model.apply(
    {'params': params},
    batch['image'],
    jnp.array([0.5, 0.5]),
    batch['text_embedding'],
    train=False
)

print(f"   Output shape: {output.shape}")
print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")

# Test with classifier-free guidance (dropout)
print("\n5. Testing with text dropout (CFG)...")
output_cfg = model.apply(
    {'params': params},
    batch['image'],
    jnp.array([0.5, 0.5]),
    batch['text_embedding'],
    train=True,
    force_drop_ids=jnp.array([1, 0]),  # Drop first, keep second
    rngs={'text_dropout': rng}
)

print(f"   CFG output shape: {output_cfg.shape}")

# Test training step simulation
print("\n6. Simulating training step...")

def get_x_t(images, eps, t):
    t = jnp.clip(t, 0, 0.99)
    return (1-t) * eps + t * images

def get_v(images, eps):
    return images - eps

# Simulate flow matching loss
eps = jax.random.normal(rng, batch['image'].shape)
t = jnp.array([0.3, 0.7])
t_full = t[:, None, None, None]
x_t = get_x_t(batch['image'], eps, t_full)
v_true = get_v(batch['image'], eps)

v_pred = model.apply(
    {'params': params},
    x_t,
    t,
    batch['text_embedding'],
    train=True,
    rngs={'text_dropout': rng}
)

loss = jnp.mean((v_pred - v_true) ** 2)
print(f"   Mock loss: {loss:.4f}")

print("\n✅ All tests passed! Text-to-image pipeline is working.")
print("\nNext steps:")
print("1. Test with real text encoder (MiniLM or CLIP)")
print("2. Test with real dataset (Conceptual Captions)")
print("3. Run full training on TPU")