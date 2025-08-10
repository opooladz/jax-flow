#!/usr/bin/env python
"""Quick test to verify HuggingFace dataset loading works."""

import sys
import numpy as np
from utils.hf_datasets import create_hf_dataloader

print("Testing HuggingFace dataset loading...")

# Test CIFAR-10
print("\n1. Testing CIFAR-10...")
loader = create_hf_dataloader(
    dataset_name="cifar10",
    batch_size=4,
    image_size=32,
    split="train",
    streaming=False,  # Don't stream for quick test
    seed=42
)

for i, batch in enumerate(loader.get_batch_iterator()):
    print(f"   Batch shape: {batch['image'].shape}, Labels: {batch['label'][:4]}")
    if i >= 1:
        break

print("   ✓ CIFAR-10 works!")

# Test TinyImageNet (streaming)
print("\n2. Testing TinyImageNet (streaming)...")
loader = create_hf_dataloader(
    dataset_name="tiny-imagenet",
    batch_size=2,
    image_size=64,
    split="train",
    streaming=True,
    seed=42
)

for i, batch in enumerate(loader.get_batch_iterator()):
    print(f"   Batch shape: {batch['image'].shape}, Labels: {batch['label'][:2]}")
    if i >= 0:  # Just one batch for streaming
        break

print("   ✓ TinyImageNet streaming works!")

print("\n✅ All tests passed! HuggingFace dataset loading is working correctly.")