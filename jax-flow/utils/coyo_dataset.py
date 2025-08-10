"""COYO-700M dataset loader with Stable Diffusion VAE and CLIP text encoder."""

import numpy as np
import jax
import jax.numpy as jnp
from datasets import load_dataset
from transformers import CLIPTokenizer, FlaxCLIPTextModel
from diffusers import FlaxAutoencoderKL
from typing import Iterator, Dict, Any, Optional
from PIL import Image
import io
import requests
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")


class COYO700MDataset:
    """COYO-700M dataset with SD-VAE encoding and CLIP text encoding."""
    
    def __init__(
        self,
        batch_size: int = 32,
        image_size: int = 256,
        streaming: bool = True,
        shuffle_buffer_size: int = 1000,
        seed: int = 42,
        use_vae: bool = True,  # Use SD-VAE for latent space
        vae_model: str = "stabilityai/sd-vae-ft-mse",  # Best VAE for training
        text_encoder: str = "openai/clip-vit-large-patch14",  # Larger CLIP
        max_text_length: int = 77,
        num_workers: int = 4,  # Parallel image downloading
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.streaming = streaming
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.use_vae = use_vae
        self.max_text_length = max_text_length
        self.num_workers = num_workers
        
        print("Initializing COYO-700M dataset...")
        
        # Initialize VAE if using latent space
        if self.use_vae:
            print(f"Loading VAE: {vae_model}")
            self.vae = FlaxAutoencoderKL.from_pretrained(vae_model, dtype=jnp.float32)
            self.vae_params = self.vae.params
            self.latent_dim = 4  # SD-VAE has 4 latent channels
            # VAE downscales by 8x
            self.latent_size = image_size // 8
            print(f"VAE loaded. Latent size: {self.latent_size}x{self.latent_size}x{self.latent_dim}")
        else:
            self.latent_dim = 3  # RGB channels
            self.latent_size = image_size
        
        # Initialize CLIP text encoder
        print(f"Loading CLIP text encoder: {text_encoder}")
        self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder)
        self.text_encoder = FlaxCLIPTextModel.from_pretrained(text_encoder)
        self.text_embed_dim = self.text_encoder.config.hidden_size
        print(f"CLIP loaded. Text embedding dim: {self.text_embed_dim}")
        
        # Thread pool for parallel image downloading
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # Load COYO-700M dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load COYO-700M dataset."""
        print("Loading COYO-700M from HuggingFace...")
        
        # COYO-700M is available in chunks, let's use a subset
        # Full dataset: "kakaobrain/coyo-700m"
        # For testing, we can use a smaller slice
        
        try:
            # Try to load a subset first
            self.dataset = load_dataset(
                "kakaobrain/coyo-700m",
                split="train",
                streaming=self.streaming,
            )
            
            if self.streaming:
                # Shuffle with buffer
                self.dataset = self.dataset.shuffle(
                    seed=self.seed,
                    buffer_size=self.shuffle_buffer_size
                )
            
            print("COYO-700M dataset loaded successfully!")
            
        except Exception as e:
            print(f"Error loading COYO-700M: {e}")
            print("Falling back to a smaller subset or mock data...")
            # Could fall back to a smaller dataset here
            raise
    
    def _download_image(self, url: str, timeout: int = 5) -> Optional[Image.Image]:
        """Download image from URL with timeout."""
        if not url or not isinstance(url, str):
            return None
        
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                return img.convert('RGB')
        except Exception:
            return None
        return None
    
    def _process_image(self, image: Image.Image) -> Optional[np.ndarray]:
        """Process image and optionally encode with VAE."""
        if image is None:
            return None
        
        # Resize to target size
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        image = np.array(image, dtype=np.float32)
        
        # Normalize to [-1, 1]
        image = (image / 127.5) - 1.0
        
        if self.use_vae:
            # Encode with VAE to latent space
            image = image[None, ...]  # Add batch dimension
            image_jax = jnp.array(image)
            
            # Encode to latent
            latent = self.vae.apply(
                {'params': self.vae_params},
                image_jax,
                method=self.vae.encode
            )
            # Get mean of distribution (ignore variance for training)
            latent = latent.latent_dist.mean
            
            # Scale by VAE scaling factor
            latent = latent * 0.18215  # SD-VAE scaling
            
            return np.array(latent[0])  # Remove batch dimension
        else:
            return image
    
    def _encode_text(self, texts: list) -> np.ndarray:
        """Encode text with CLIP."""
        # Tokenize
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="jax"
        )
        
        # Encode
        outputs = self.text_encoder(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask']
        )
        
        # Use pooled output
        text_embeddings = outputs.pooler_output
        
        return np.array(text_embeddings)
    
    def _process_batch(self, examples: Dict[str, list]) -> Optional[Dict[str, Any]]:
        """Process a batch of COYO examples."""
        # COYO fields: 'url', 'text', 'width', 'height', 'similarity', etc.
        
        urls = examples.get('url', [])
        texts = examples.get('text', examples.get('alt', []))
        
        if not urls or not texts:
            return None
        
        # Download images in parallel
        with self.executor as executor:
            images = list(executor.map(self._download_image, urls))
        
        # Process valid images
        processed_images = []
        processed_texts = []
        
        for img, text in zip(images, texts):
            if img is not None:
                processed = self._process_image(img)
                if processed is not None:
                    processed_images.append(processed)
                    processed_texts.append(text if text else "An image")
        
        if len(processed_images) == 0:
            return None
        
        # Stack images
        images_array = np.stack(processed_images, axis=0)
        
        # Encode texts
        text_embeddings = self._encode_text(processed_texts)
        
        return {
            'image': images_array,  # Either latents or pixels
            'text_embedding': text_embeddings,
            'text': processed_texts,
        }
    
    def get_batch_iterator(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """Iterate over batches."""
        batch_buffer = []
        
        for example in self.dataset:
            batch_buffer.append(example)
            
            # Process when we have enough examples
            if len(batch_buffer) >= self.batch_size * 2:  # Get extra in case some fail
                batch_dict = {
                    k: [ex.get(k) for ex in batch_buffer]
                    for k in batch_buffer[0].keys()
                }
                
                processed = self._process_batch(batch_dict)
                
                if processed and processed['image'].shape[0] >= self.batch_size:
                    # Yield exactly batch_size
                    yield {
                        'image': jnp.array(processed['image'][:self.batch_size]),
                        'text_embedding': jnp.array(processed['text_embedding'][:self.batch_size]),
                        'text': processed['text'][:self.batch_size],
                    }
                    batch_buffer = []  # Clear buffer after successful batch
                
                # If we couldn't get a full batch, keep collecting
    
    def create_infinite_iterator(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """Create infinite iterator."""
        while True:
            for batch in self.get_batch_iterator():
                yield batch
            # Reload dataset for next epoch
            if self.streaming:
                self._load_dataset()


def create_coyo_dataloader(
    batch_size: int = 32,
    image_size: int = 256,
    use_vae: bool = True,
    streaming: bool = True,
) -> COYO700MDataset:
    """Create COYO-700M dataloader.
    
    Args:
        batch_size: Batch size
        image_size: Target image size (before VAE encoding)
        use_vae: Whether to use SD-VAE for latent space
        streaming: Whether to stream the dataset
    
    Returns:
        COYO700MDataset instance
    """
    return COYO700MDataset(
        batch_size=batch_size,
        image_size=image_size,
        use_vae=use_vae,
        streaming=streaming,
    )