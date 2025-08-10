"""HuggingFace dataset streaming support."""

import numpy as np
import jax
import jax.numpy as jnp
from datasets import load_dataset
from PIL import Image
import io
from typing import Iterator, Dict, Any, Optional


class HFDatasetStreamer:
    """Streaming dataset loader for HuggingFace datasets."""
    
    def __init__(
        self, 
        dataset_name: str,
        split: str = "train",
        batch_size: int = 32,
        image_size: int = 64,
        num_classes: Optional[int] = None,
        streaming: bool = True,
        shuffle_buffer_size: int = 10000,
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.streaming = streaming
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Get number of classes if not provided
        if self.num_classes is None:
            self.num_classes = self._get_num_classes()
    
    def _load_dataset(self):
        """Load the HuggingFace dataset."""
        if self.dataset_name == "tiny-imagenet":
            # Use the Maysee/tiny-imagenet dataset from HuggingFace
            dataset = load_dataset(
                "Maysee/tiny-imagenet", 
                split=self.split,
                streaming=self.streaming
            )
        elif self.dataset_name == "cifar10":
            # CIFAR-10 dataset
            dataset = load_dataset(
                "cifar10", 
                split=self.split,
                streaming=self.streaming
            )
        elif self.dataset_name == "cifar100":
            # CIFAR-100 dataset
            dataset = load_dataset(
                "cifar100", 
                split=self.split,
                streaming=self.streaming
            )
        else:
            # Generic loading for other datasets
            dataset = load_dataset(
                self.dataset_name, 
                split=self.split,
                streaming=self.streaming
            )
        
        if self.streaming:
            # Shuffle with buffer for streaming
            dataset = dataset.shuffle(
                seed=self.seed,
                buffer_size=self.shuffle_buffer_size
            )
        else:
            # Regular shuffle for non-streaming
            dataset = dataset.shuffle(seed=self.seed)
        
        return dataset
    
    def _get_num_classes(self) -> int:
        """Get number of classes from dataset."""
        if self.dataset_name == "tiny-imagenet":
            return 200  # TinyImageNet has 200 classes
        elif self.dataset_name == "cifar10":
            return 10  # CIFAR-10 has 10 classes
        elif self.dataset_name == "cifar100":
            return 100  # CIFAR-100 has 100 classes
        
        # For other datasets, try to infer from features
        if hasattr(self.dataset, 'features'):
            if 'label' in self.dataset.features:
                if hasattr(self.dataset.features['label'], 'num_classes'):
                    return self.dataset.features['label'].num_classes
            elif 'fine_label' in self.dataset.features:
                if hasattr(self.dataset.features['fine_label'], 'num_classes'):
                    return self.dataset.features['fine_label'].num_classes
        
        # Default fallback
        return 1000
    
    def _preprocess_image(self, image: Any) -> np.ndarray:
        """Preprocess image to numpy array."""
        from PIL import Image
        
        # Handle PIL Image
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            image = np.array(image, dtype=np.float32)
        # Handle numpy array
        elif isinstance(image, np.ndarray):
            if image.shape[-1] != 3:
                # Convert grayscale to RGB
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
            # Resize if needed
            if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
                img = Image.fromarray(image.astype(np.uint8))
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                image = np.array(img, dtype=np.float32)
        
        # Normalize to [-1, 1]
        image = image / 127.5 - 1.0
        
        return image
    
    def _process_batch(self, examples: Dict[str, list]) -> Dict[str, np.ndarray]:
        """Process a batch of examples."""
        # Get images and labels
        images = []
        labels = []
        
        # Determine the image field name
        image_field = 'image' if 'image' in examples else 'img'
        
        for i in range(len(examples[image_field])):
            img = self._preprocess_image(examples[image_field][i])
            images.append(img)
            
            # Handle label field variations
            if 'label' in examples:
                labels.append(examples['label'][i])
            elif 'labels' in examples:
                labels.append(examples['labels'][i])
            elif 'fine_label' in examples:  # CIFAR-100
                labels.append(examples['fine_label'][i])
            else:
                labels.append(0)  # Default label if not found
        
        images = np.stack(images, axis=0)
        labels = np.array(labels, dtype=np.int32)
        
        return {
            'image': images,
            'label': labels
        }
    
    def get_batch_iterator(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """Get an iterator that yields batches of data."""
        if self.streaming:
            # For streaming datasets
            batch = []
            for example in self.dataset:
                batch.append(example)
                if len(batch) == self.batch_size:
                    # Process and yield batch
                    batch_dict = {k: [ex[k] for ex in batch] for k in batch[0].keys()}
                    processed = self._process_batch(batch_dict)
                    yield {
                        'image': jnp.array(processed['image']),
                        'label': jnp.array(processed['label'])
                    }
                    batch = []
        else:
            # For non-streaming datasets
            dataset = self.dataset.batch(self.batch_size)
            for batch in dataset:
                processed = self._process_batch(batch)
                yield {
                    'image': jnp.array(processed['image']),
                    'label': jnp.array(processed['label'])
                }
    
    def create_infinite_iterator(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """Create an infinite iterator that cycles through the dataset."""
        while True:
            for batch in self.get_batch_iterator():
                yield batch
            
            # Re-shuffle for next epoch if streaming
            if self.streaming:
                self.dataset = self._load_dataset()


def create_hf_dataloader(
    dataset_name: str,
    batch_size: int,
    image_size: int = 64,
    split: str = "train",
    streaming: bool = True,
    seed: int = 42
) -> HFDatasetStreamer:
    """Create a HuggingFace dataset streamer.
    
    Args:
        dataset_name: Name of the dataset (e.g., "tiny-imagenet")
        batch_size: Batch size for loading
        image_size: Target image size (will resize if needed)
        split: Dataset split to use
        streaming: Whether to use streaming mode
        seed: Random seed for shuffling
    
    Returns:
        HFDatasetStreamer instance
    """
    return HFDatasetStreamer(
        dataset_name=dataset_name,
        split=split,
        batch_size=batch_size,
        image_size=image_size,
        streaming=streaming,
        seed=seed
    )