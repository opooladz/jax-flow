"""Mock text-to-image dataset for testing without downloading models."""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Iterator, Dict, Any, Optional
from PIL import Image


class MockTextImageDataset:
    """Mock dataset for testing text-to-image pipeline without real models."""
    
    def __init__(
        self, 
        batch_size: int = 4,
        image_size: int = 64,
        text_embed_dim: int = 384,  # Smaller embedding dimension
        num_samples: int = 100,  # Fixed number of mock samples
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.text_embed_dim = text_embed_dim
        self.num_samples = num_samples
        
        # Generate mock data
        np.random.seed(seed)
        
        # Mock images (random patterns)
        self.images = []
        for i in range(num_samples):
            # Create simple patterns for visual verification
            img = np.zeros((image_size, image_size, 3), dtype=np.float32)
            
            # Different patterns based on index
            if i % 4 == 0:  # Horizontal stripes
                for y in range(0, image_size, 8):
                    img[y:y+4] = np.random.rand(3) * 2 - 1
            elif i % 4 == 1:  # Vertical stripes
                for x in range(0, image_size, 8):
                    img[:, x:x+4] = np.random.rand(3) * 2 - 1
            elif i % 4 == 2:  # Checkerboard
                for y in range(0, image_size, 16):
                    for x in range(0, image_size, 16):
                        if (x//16 + y//16) % 2 == 0:
                            img[y:y+16, x:x+16] = np.random.rand(3) * 2 - 1
            else:  # Random noise
                img = np.random.randn(image_size, image_size, 3) * 0.5
            
            # Normalize to [-1, 1]
            img = np.clip(img, -1, 1)
            self.images.append(img)
        
        # Mock text embeddings (random but consistent)
        self.text_embeddings = np.random.randn(num_samples, text_embed_dim).astype(np.float32)
        self.text_embeddings /= np.linalg.norm(self.text_embeddings, axis=1, keepdims=True)
        
        # Mock text descriptions
        patterns = ["horizontal stripes", "vertical stripes", "checkerboard", "random noise"]
        self.texts = [f"A synthetic image with {patterns[i % 4]} pattern #{i}" for i in range(num_samples)]
        
        print(f"Created mock dataset with {num_samples} samples")
        print(f"Image size: {image_size}x{image_size}")
        print(f"Text embedding dimension: {text_embed_dim}")
    
    def get_batch_iterator(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """Get an iterator that yields batches of mock data."""
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            if len(batch_indices) < self.batch_size:
                # Pad last batch if needed
                batch_indices = np.concatenate([
                    batch_indices,
                    indices[:self.batch_size - len(batch_indices)]
                ])
            
            batch_images = np.stack([self.images[idx] for idx in batch_indices])
            batch_text_emb = np.stack([self.text_embeddings[idx] for idx in batch_indices])
            batch_texts = [self.texts[idx] for idx in batch_indices]
            
            yield {
                'image': jnp.array(batch_images),
                'text_embedding': jnp.array(batch_text_emb),
                'text': batch_texts
            }
    
    def create_infinite_iterator(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """Create an infinite iterator that cycles through the dataset."""
        while True:
            for batch in self.get_batch_iterator():
                yield batch


def create_mock_dataloader(
    batch_size: int = 4,
    image_size: int = 64,
    text_embed_dim: int = 384,
) -> MockTextImageDataset:
    """Create a mock text-to-image dataset for testing.
    
    Args:
        batch_size: Batch size for loading
        image_size: Size of mock images
        text_embed_dim: Dimension of mock text embeddings
    
    Returns:
        MockTextImageDataset instance
    """
    return MockTextImageDataset(
        batch_size=batch_size,
        image_size=image_size,
        text_embed_dim=text_embed_dim,
    )