# Text-to-Image with COYO-700M

## Overview

This implementation supports text-to-image generation using flow matching with:
- **COYO-700M** dataset (700 million image-text pairs)
- **Stable Diffusion VAE** for latent space training (recommended)
- **CLIP** text encoder (768-dim embeddings)

## Why Latent Space (VAE)?

### Pixel Space (No VAE)
- **Size**: 256×256×3 = 196,608 dimensions
- **Pros**: Simple, direct
- **Cons**: Slow, memory-intensive, harder to train

### Latent Space (With SD-VAE)
- **Size**: 32×32×4 = 4,096 dimensions (48x smaller!)
- **Pros**: 
  - Much faster training
  - Less memory usage
  - Better quality (proven by Stable Diffusion)
  - Can generate at higher resolutions later
- **Cons**: Need VAE encoder/decoder

**Recommendation**: Use VAE latent space for COYO-700M

## Quick Start

### 1. Test Setup (CPU/Small GPU)
```bash
# Quick test with mock data
python test_text_mock.py

# Test with COYO (downloads data)
python train_coyo_vae.py --test
```

### 2. Full Training (TPU v3-8)
```bash
# Full COYO-700M training with VAE
python train_coyo_vae.py

# Or manually configure:
python train_text_to_image.py \
  --dataset_name coyo \
  --batch_size 256 \
  --image_size 256 \
  --model.use_stable_vae 1 \
  --model.patch_size 2 \
  --model.preset large \
  --max_steps 500000
```

## Model Configurations

### VAE Latent Space Settings
For 256×256 images → 32×32×4 latents:
- `patch_size`: 2 (for 32×32 latents)
- `hidden_size`: 1024 (large model)
- `depth`: 24
- `num_heads`: 16

### Pixel Space Settings
For 256×256×3 direct pixels:
- `patch_size`: 8 (for 32×32 patches)
- `hidden_size`: 1024
- `depth`: 24
- `num_heads`: 16

## Dataset Options

### COYO-700M (Recommended)
- **Size**: 700M image-text pairs
- **Quality**: High quality, filtered
- **Languages**: Multilingual
- **Access**: Streaming from HuggingFace

### Alternatives
- **LAION-2B**: Even larger, 2 billion pairs
- **LAION-Art**: Subset focused on artistic images
- **Conceptual Captions 12M**: Smaller, good for testing

## Text Encoders

Currently using:
- **CLIP ViT-L/14**: 768-dim embeddings, 77 tokens max

Future options:
- **T5-XXL**: Better for long descriptions
- **CLIP + T5**: Best quality (like SDXL)

## Memory Requirements

### With VAE (Recommended)
- **TPU v3-8**: Batch size 256-512
- **GPU (A100)**: Batch size 32-64
- **GPU (V100)**: Batch size 16-32

### Without VAE
- **TPU v3-8**: Batch size 64-128
- **GPU (A100)**: Batch size 8-16
- **GPU (V100)**: Batch size 4-8

## Training Tips

1. **Start with VAE**: Much more efficient
2. **Use mixed precision**: Add `--use_amp 1` for faster training
3. **Gradient accumulation**: If batch size is too small
4. **Classifier-free guidance**: Keep `text_dropout_prob=0.1`
5. **Learning rate**: 1e-4 for large batches, 5e-5 for small

## Inference

After training, generate images with:
```python
# Load model
model = load_checkpoint("path/to/checkpoint")

# Generate with text prompt
prompt = "A beautiful sunset over mountains"
image = model.generate(prompt, cfg_scale=7.5)

# Decode from latent if using VAE
if use_vae:
    image = vae.decode(image)
```

## Expected Results

With proper training:
- **10k steps**: Basic shapes and colors
- **50k steps**: Recognizable objects
- **100k steps**: Good quality
- **500k steps**: High quality
- **1M+ steps**: State-of-the-art

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use gradient checkpointing
- Use VAE (if not already)
- Reduce model size

### Slow Training
- Ensure using VAE
- Check data loading isn't bottleneck
- Use larger batch size
- Use mixed precision

### Poor Quality
- Train longer
- Increase CFG scale during inference
- Check text encoder is working
- Verify data quality