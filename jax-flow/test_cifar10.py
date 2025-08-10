#!/usr/bin/env python
"""Test CIFAR-10 loading which should be faster."""

import sys
from datasets import load_dataset
import numpy as np

def test_cifar10():
    """Test CIFAR-10 dataset loading."""
    print("Testing CIFAR-10 dataset loading...")
    
    # Load CIFAR-10 which is smaller and faster
    print("Loading cifar10...")
    dataset = load_dataset(
        "cifar10", 
        split="train",
        streaming=False  # Don't stream for CIFAR-10 since it's small
    )
    
    print(f"Dataset loaded! Total examples: {len(dataset)}")
    
    # Check first example
    example = dataset[0]
    print(f"\nFirst example keys: {example.keys()}")
    
    if 'img' in example:
        img = np.array(example['img'])
        print(f"  Image shape: {img.shape}")
        print(f"  Image dtype: {img.dtype}")
    
    if 'label' in example:
        print(f"  Label: {example['label']}")
    
    print("\nTest completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_cifar10()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)