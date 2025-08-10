# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a JAX implementation of flow-matching models (Rectified Flow), which are a type of generative model similar to diffusion models. The implementation is based on the Diffusion Transformer (DiT) architecture and supports training on image datasets like ImageNet256 and CelebAHQ256.

## Key Commands

### Training

Train flow-matching models with different configurations:

#### HuggingFace Datasets (New)
```bash
# Train on CIFAR-10
python train_flow.py --dataset_name cifar10 --batch_size 128 --model.preset big --model.patch_size 8 --model.use_stable_vae 0

# Train on TinyImageNet  
python train_flow.py --dataset_name tiny-imagenet --batch_size 64 --model.preset big --model.patch_size 8 --model.use_stable_vae 0

# Or use the provided script
./train_tiny_imagenet.sh
```

#### Original TensorFlow Datasets

```bash
# DiT-B on ImageNet256 with Stable Diffusion VAE (latent space)
python train_flow.py --dataset_name imagenet256 --wandb.name DiT-B --model.depth 12 --model.hidden_size 768 --model.patch_size 2 --model.num_heads 16 --model.mlp_ratio 4 --batch_size 512

# DiT-B on CelebAHQ256 with Stable Diffusion VAE (latent space)
python train_flow.py --dataset_name celebahq256 --wandb.name DiT-B-CelebA --model.depth 12 --model.hidden_size 768 --model.patch_size 2 --model.num_heads 16 --model.mlp_ratio 4 --batch_size 512

# DiT-B on CelebAHQ256 in pixel space (no VAE)
python train_flow.py --dataset_name celebahq256 --wandb.name DiT-B-CelebAPixel --model.depth 12 --model.hidden_size 768 --model.patch_size 8 --model.num_heads 16 --model.mlp_ratio 4 --batch_size 512 --use_stable_vae 0
```

### Evaluation

Evaluate FID (Fréchet Inception Distance) on trained models:

```bash
python eval_fid.py --load_dir [checkpoint_dir] --dataset_name imagenet256 --fid_stats data/imagenet256_fidstats_openai.npz --cfg_scale 4 --denoise_timesteps 500
```

## Architecture Overview

### Core Components

1. **Flow Matching Model** (`train_flow.py`)
   - Implements the flow-matching training objective where the model learns to predict velocities between noise and data
   - Uses linear interpolation: `x_t = (1-t) * x_0 + t * x_1`
   - Trains to match normalized velocity: `v_theta(x_t) <- (x_1 - x_0)`

2. **Diffusion Transformer** (`diffusion_transformer.py`)
   - DiT (Diffusion Transformer) backbone architecture
   - Includes timestep embedding, class label embedding with dropout for classifier-free guidance
   - Supports multiple model presets: debug, big, semilarge, large, xlarge

3. **VAE Integration** (`utils/stable_vae.py`)
   - Optional Stable Diffusion VAE encoder/decoder for latent space diffusion
   - Can train in either pixel space or VAE latent space

4. **Training Infrastructure** (`utils/`)
   - `train_state.py`: Training state management with Flax
   - `checkpoint.py`: Model checkpointing
   - `wandb.py`: Weights & Biases integration for experiment tracking
   - `fid.py`: FID computation for model evaluation

### Key Configuration Parameters

- **Model Architecture**: `depth`, `hidden_size`, `patch_size`, `num_heads`, `mlp_ratio`
- **Training**: `lr` (0.0001), `beta1` (0.9), `beta2` (0.99), `batch_size`
- **Flow Matching**: `denoise_timesteps`, `t_sampler` (log-normal), `t_conditioning`
- **Classifier-Free Guidance**: `class_dropout_prob` (0.1), `cfg_scale` (4.0)
- **VAE**: `use_stable_vae` (0 or 1)

### Dataset Support

#### HuggingFace Datasets (Streaming)
The codebase now supports HuggingFace datasets with streaming capabilities:
- **CIFAR-10**: 10 classes, 32x32 images
- **CIFAR-100**: 100 classes, 32x32 images  
- **TinyImageNet**: 200 classes, 64x64 images (uses Maysee/tiny-imagenet)

These datasets stream directly from HuggingFace Hub without requiring local storage.

#### TensorFlow Datasets
Original support for compiled TFDS datasets for `imagenet2012` or `celebahq`. These datasets need to be prepared using the tfds_builders repository mentioned in the README.

### Model Presets

The code includes predefined model configurations:
- **debug**: Minimal model for testing (hidden_size=64, depth=2)
- **big**: DiT-B equivalent (hidden_size=768, depth=12)
- **semilarge/large**: DiT-L variants (hidden_size=1024, depth=22-24)
- **xlarge**: DiT-XL (hidden_size=1152, depth=28)

### Sampling Process

During inference, the model solves an ODE using Euler sampling:
```python
x = noise
for i in range(N):
    dt = 1 / N
    x = x + v_theta(x) * dt
```

The number of denoising steps can be dynamically adjusted (deterministic sampling).