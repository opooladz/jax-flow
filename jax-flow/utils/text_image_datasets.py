"""Text-to-image dataset support using CLIP text encoder."""

import numpy as np
import jax
import jax.numpy as jnp
from datasets import load_dataset
from transformers import CLIPTokenizer, FlaxCLIPTextModel
from typing import Iterator, Dict, Any, Optional
from PIL import Image
import io
import requests


class TextImageDatasetStreamer:
    """Streaming dataset loader for text-to-image datasets."""
    
    def __init__(
        self, 
        dataset_name: str,
        split: str = "train",
        batch_size: int = 32,
        image_size: int = 256,
        streaming: bool = True,
        shuffle_buffer_size: int = 1000,
        seed: int = 42,
        text_encoder_name: str = "openai/clip-vit-base-patch32",
        max_text_length: int = 77,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.image_size = image_size
        self.streaming = streaming
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.max_text_length = max_text_length
        
        # Initialize CLIP text encoder
        print(f"Loading CLIP text encoder: {text_encoder_name}")
        self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder_name)
        self.text_encoder = FlaxCLIPTextModel.from_pretrained(text_encoder_name)
        
        # Load dataset
        self.dataset = self._load_dataset()
    
    def _load_dataset(self):
        """Load the text-to-image dataset."""
        print(f"Loading dataset: {self.dataset_name}")
        
        if self.dataset_name == "conceptual_captions":
            # Conceptual Captions (smaller, good for testing)
            dataset = load_dataset(
                "google-research-datasets/conceptual_captions",
                "unlabeled",  # Use the unlabeled split
                split=self.split,
                streaming=self.streaming,
                trust_remote_code=True
            )
        elif self.dataset_name == "laion-art":
            # LAION-Art subset
            dataset = load_dataset(
                "laion/laion-art", 
                split=self.split,
                streaming=self.streaming
            )
        elif self.dataset_name == "coyo-tiny":
            # Small COYO subset for testing
            dataset = load_dataset(
                "kakaobrain/coyo-700m",
                split=self.split + "[:10000]",  # Just first 10k for testing
                streaming=False  # Can't stream with slice
            )
        else:
            # Generic loading
            dataset = load_dataset(
                self.dataset_name, 
                split=self.split,
                streaming=self.streaming
            )
        
        if self.streaming:
            dataset = dataset.shuffle(
                seed=self.seed,
                buffer_size=self.shuffle_buffer_size
            )
        else:
            dataset = dataset.shuffle(seed=self.seed)
        
        return dataset
    
    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                return img
        except:
            pass
        return None
    
    def _preprocess_image(self, image: Any) -> Optional[np.ndarray]:
        """Preprocess image to numpy array."""
        if image is None:
            return None
            
        # If it's a URL string, download it
        if isinstance(image, str):
            image = self._download_image(image)
            if image is None:
                return None
        
        # Convert to PIL Image if needed
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            image = np.array(image, dtype=np.float32)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
                img = Image.fromarray(image.astype(np.uint8))
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                image = np.array(img, dtype=np.float32)
        else:
            return None
        
        # Normalize to [-1, 1]
        image = image / 127.5 - 1.0
        return image
    
    def _encode_text(self, texts: list) -> np.ndarray:
        """Encode text captions to embeddings using CLIP."""
        # Tokenize
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="jax"
        )
        
        # Get text embeddings
        outputs = self.text_encoder(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask']
        )
        
        # Use pooled output (final [CLS] token)
        text_embeddings = outputs.pooler_output
        
        return np.array(text_embeddings)
    
    def _process_batch(self, examples: Dict[str, list]) -> Optional[Dict[str, np.ndarray]]:
        """Process a batch of image-text pairs."""
        images = []
        texts = []
        
        # Handle different field names across datasets
        image_field = None
        text_field = None
        
        # Find image field
        for field in ['image', 'url', 'image_url', 'URL']:
            if field in examples:
                image_field = field
                break
        
        # Find text field
        for field in ['caption', 'text', 'alt_text', 'TEXT']:
            if field in examples:
                text_field = field
                break
        
        if not image_field or not text_field:
            return None
        
        # Process each example
        valid_indices = []
        for i in range(len(examples[image_field])):
            # Process image
            img = self._preprocess_image(examples[image_field][i])
            if img is not None:
                images.append(img)
                # Get text
                text = examples[text_field][i] if text_field in examples else ""
                texts.append(text)
                valid_indices.append(i)
        
        if len(images) == 0:
            return None
        
        images = np.stack(images, axis=0)
        text_embeddings = self._encode_text(texts)
        
        return {
            'image': images,
            'text_embedding': text_embeddings,
            'text': texts  # Keep raw text for logging
        }
    
    def get_batch_iterator(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """Get an iterator that yields batches of data."""
        if self.streaming:
            batch = []
            for example in self.dataset:
                batch.append(example)
                if len(batch) >= self.batch_size * 2:  # Collect more in case some fail
                    batch_dict = {k: [ex.get(k) for ex in batch] for k in batch[0].keys()}
                    processed = self._process_batch(batch_dict)
                    if processed and processed['image'].shape[0] >= self.batch_size:
                        # Take only batch_size samples
                        yield {
                            'image': jnp.array(processed['image'][:self.batch_size]),
                            'text_embedding': jnp.array(processed['text_embedding'][:self.batch_size]),
                            'text': processed['text'][:self.batch_size]
                        }
                        batch = []
                    elif processed and processed['image'].shape[0] > 0:
                        # Keep collecting if we don't have enough
                        continue
                    else:
                        # Failed batch, try again
                        batch = []
        else:
            dataset = self.dataset.batch(self.batch_size * 2)  # Batch more in case some fail
            for batch in dataset:
                processed = self._process_batch(batch)
                if processed and processed['image'].shape[0] >= self.batch_size:
                    yield {
                        'image': jnp.array(processed['image'][:self.batch_size]),
                        'text_embedding': jnp.array(processed['text_embedding'][:self.batch_size]),
                        'text': processed['text'][:self.batch_size]
                    }
    
    def create_infinite_iterator(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """Create an infinite iterator that cycles through the dataset."""
        while True:
            for batch in self.get_batch_iterator():
                yield batch
            
            # Re-shuffle for next epoch if streaming
            if self.streaming:
                self.dataset = self._load_dataset()


def create_text_image_dataloader(
    dataset_name: str,
    batch_size: int,
    image_size: int = 256,
    split: str = "train",
    streaming: bool = True,
    seed: int = 42
) -> TextImageDatasetStreamer:
    """Create a text-to-image dataset streamer.
    
    Args:
        dataset_name: Name of the dataset
        batch_size: Batch size for loading
        image_size: Target image size
        split: Dataset split to use
        streaming: Whether to use streaming mode
        seed: Random seed for shuffling
    
    Returns:
        TextImageDatasetStreamer instance
    """
    return TextImageDatasetStreamer(
        dataset_name=dataset_name,
        split=split,
        batch_size=batch_size,
        image_size=image_size,
        streaming=streaming,
        seed=seed
    )