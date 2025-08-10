#!/usr/bin/env python
"""Test COYO data loading speed."""

import time
from utils.coyo_dataset import create_coyo_dataloader

print("Testing COYO-700M data loading speed...")
print("This will download images from the web, so speed depends on network")

# Create minimal loader
loader = create_coyo_dataloader(
    batch_size=8,  # Small batch for testing
    image_size=256,
    use_vae=True,
    streaming=True,
)

print("\nCreating iterator...")
iterator = loader.create_infinite_iterator()

print("\nFetching first batch (this downloads images from web)...")
start = time.time()
batch = next(iterator)
elapsed = time.time() - start

print(f"\nFirst batch took: {elapsed:.1f} seconds")
print(f"Batch shapes:")
print(f"  Images: {batch['image'].shape}")
print(f"  Text embeddings: {batch['text_embedding'].shape}")

if elapsed > 60:
    print(f"\n⚠️  Data loading is slow ({elapsed:.1f}s for batch of 8)")
    print("This is normal for COYO as images are downloaded from the web")
    print("Training will be slow until the shuffle buffer fills up")
else:
    print(f"\n✓ Data loading speed is reasonable")

print("\nFetching second batch (should be faster)...")
start = time.time()
batch = next(iterator)
elapsed2 = time.time() - start
print(f"Second batch took: {elapsed2:.1f} seconds")

if elapsed2 < elapsed / 2:
    print("✓ Subsequent batches are faster (buffer is working)")