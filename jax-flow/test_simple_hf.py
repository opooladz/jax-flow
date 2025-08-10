#!/usr/bin/env python
"""Simple test for HuggingFace dataset loading."""

import sys
from datasets import load_dataset

def test_simple():
    """Test basic dataset access."""
    print("Testing basic HuggingFace dataset access...")
    
    # Try to load the dataset
    print("Loading Maysee/tiny-imagenet...")
    dataset = load_dataset(
        "Maysee/tiny-imagenet", 
        split="train",
        streaming=True
    )
    
    print("Dataset loaded successfully!")
    
    # Get first example
    print("\nGetting first example...")
    for i, example in enumerate(dataset):
        print(f"Example {i+1}:")
        print(f"  Keys: {example.keys()}")
        if 'image' in example:
            print(f"  Image type: {type(example['image'])}")
        if 'label' in example:
            print(f"  Label: {example['label']}")
        
        if i >= 2:  # Just check a few
            break
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    try:
        test_simple()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)