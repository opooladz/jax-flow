"""HuggingFace text-to-image dataset support for COYO-700M and similar datasets."""

import numpy as np
import jax
import jax.numpy as jnp
from datasets import load_dataset
from transformers import AutoTokenizer, FlaxT5Model, T5Tokenizer
from typing import Iterator, Dict, Any, Optional
import torch
from PIL import Image


class TextImageDatasetStreamer:
    """Streaming dataset loader for text-to-image datasets like COYO-700M."""
    
    def __init__(
        self, 
        dataset_name: str,
        split: str = "train",
        batch_size: int = 32,
        image_size: int = 256,
        streaming: bool = True,
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
        text_encoder: str = "t5-base",
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
        
        # Initialize text encoder
        self.tokenizer = T5Tokenizer.from_pretrained(text_encoder)
        self.text_encoder = FlaxT5Model.from_pretrained(text_encoder)
        
        # Load dataset
        self.dataset = self._load_dataset()
    
    def _load_dataset(self):
        """Load the text-to-image dataset."""
        if self.dataset_name == "coyo-700m":
            # COYO-700M subset
            dataset = load_dataset(
                "kakaobrain/coyo-700m", 
                split=self.split,
                streaming=self.streaming
            )
        elif self.dataset_name == "laion-art":
            # LAION-Art subset
            dataset = load_dataset(
                "laion/laion-art", 
                split=self.split,
                streaming=self.streaming
            )
        elif self.dataset_name == "cc12m":
            # Conceptual Captions 12M
            dataset = load_dataset(
                "conceptual_12m", 
                split=self.split,
                streaming=self.streaming
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
    
    def _preprocess_image(self, image: Any) -> np.ndarray:
        """Preprocess image to numpy array."""
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            image = np.array(image, dtype=np.float32)
        elif isinstance(image, np.ndarray):
            if image.shape[-1] != 3:
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
            if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
                img = Image.fromarray(image.astype(np.uint8))
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                image = np.array(img, dtype=np.float32)
        
        # Normalize to [-1, 1]
        image = image / 127.5 - 1.0
        return image
    
    def _encode_text(self, texts: list) -> np.ndarray:
        """Encode text captions to embeddings."""
        # Tokenize
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="np"
        )
        
        # Get text embeddings
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask']
            )
            # Use pooled output or last hidden state
            text_embeddings = outputs.last_hidden_state.mean(axis=1)  # Simple pooling
        
        return text_embeddings.numpy()
    
    def _process_batch(self, examples: Dict[str, list]) -> Dict[str, np.ndarray]:
        """Process a batch of image-text pairs."""
        images = []
        texts = []
        
        # Different datasets have different field names
        image_field = 'image' if 'image' in examples else 'url'  # COYO uses 'url'
        text_field = 'text' if 'text' in examples else 'caption'
        
        for i in range(len(examples[image_field])):
            # Process image
            img = self._preprocess_image(examples[image_field][i])
            images.append(img)
            
            # Get text
            text = examples[text_field][i] if text_field in examples else ""
            texts.append(text)
        
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
                if len(batch) == self.batch_size:
                    batch_dict = {k: [ex[k] for ex in batch] for k in batch[0].keys()}
                    processed = self._process_batch(batch_dict)
                    yield {
                        'image': jnp.array(processed['image']),
                        'text_embedding': jnp.array(processed['text_embedding']),
                        'text': processed['text']
                    }
                    batch = []
        else:
            dataset = self.dataset.batch(self.batch_size)
            for batch in dataset:
                processed = self._process_batch(batch)
                yield {
                    'image': jnp.array(processed['image']),
                    'text_embedding': jnp.array(processed['text_embedding']),
                    'text': processed['text']
                }


# Example text conditioning module for DiT
def create_text_conditioned_dit_embedder():
    """Create a text embedding module for DiT."""
    import flax.linen as nn
    
    class TextEmbedder(nn.Module):
        """Text embedder for conditioning DiT on text embeddings."""
        hidden_size: int
        dropout_prob: float = 0.1
        
        @nn.compact
        def __call__(self, text_embeddings, train=True):
            # Project text embeddings to model dimension
            x = nn.Dense(self.hidden_size)(text_embeddings)
            x = nn.silu(x)
            x = nn.Dense(self.hidden_size)(x)
            
            # Dropout for classifier-free guidance
            if train and self.dropout_prob > 0:
                rng = self.make_rng('dropout')
                drop_mask = jax.random.bernoulli(rng, self.dropout_prob, (text_embeddings.shape[0],))
                # Zero out embeddings for dropped samples
                x = jnp.where(drop_mask[:, None], jnp.zeros_like(x), x)
            
            return x
    
    return TextEmbedder