#!/usr/bin/env python
"""Test HuggingFace dataset loading."""

import sys
import numpy as np
from utils.hf_datasets import create_hf_dataloader

def test_dataset_loading():
    """Test loading TinyImageNet from HuggingFace."""
    print("Testing HuggingFace dataset loading...")
    
    # Create dataloader
    loader = create_hf_dataloader(
        dataset_name="tiny-imagenet",
        batch_size=4,
        image_size=64,
        split="train",
        streaming=True,
        seed=42
    )
    
    print(f"Dataset: tiny-imagenet")
    print(f"Number of classes: {loader.num_classes}")
    
    # Get a batch
    print("\nGetting first batch...")
    for i, batch in enumerate(loader.get_batch_iterator()):
        print(f"Batch {i+1}:")
        print(f"  Images shape: {batch['image'].shape}")
        print(f"  Labels shape: {batch['label'].shape}")
        print(f"  Image dtype: {batch['image'].dtype}")
        print(f"  Label dtype: {batch['label'].dtype}")
        print(f"  Image range: [{batch['image'].min():.2f}, {batch['image'].max():.2f}]")
        print(f"  Sample labels: {batch['label'][:4]}")
        
        if i >= 2:  # Just test a few batches
            break
    
    print("\nDataset loading test successful!")
    return True

if __name__ == "__main__":
    try:
        test_dataset_loading()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)