# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX implementation of flow-matching models (Rectified Flow) - generative models similar to diffusion models but with straight trajectories between noise and data. Built on the Diffusion Transformer (DiT) architecture with support for both pixel-space and latent-space (VAE) training.

## Key Commands

### Training

```bash
# Quick test with debug preset
python train_flow.py --dataset_name cifar10 --model.preset debug --max_steps 100

# HuggingFace Datasets (streaming, no storage needed)
python train_flow.py --dataset_name cifar10 --batch_size 128 --model.preset big --model.patch_size 8 --model.use_stable_vae 0
python train_flow.py --dataset_name cifar100 --batch_size 128 --model.preset big --model.patch_size 8 --model.use_stable_vae 0
python train_flow.py --dataset_name tiny-imagenet --batch_size 64 --model.preset big --model.patch_size 8 --model.use_stable_vae 0

# TensorFlow Datasets (requires TFDS preparation)
python train_flow.py --dataset_name imagenet256 --model.preset big --batch_size 512 --model.patch_size 2
python train_flow.py --dataset_name celebahq256 --model.preset big --batch_size 512 --model.patch_size 2

# TPU training
./setup_tpu.sh  # One-time setup
./train_tpu.sh  # Run training with optimized settings
```

### Evaluation

```bash
# FID evaluation
python eval_fid.py --load_dir [checkpoint_dir] --dataset_name [dataset] --cfg_scale 4 --denoise_timesteps 500

# Test dataset loading
python quick_test.py      # Test HF dataset streaming
python test_cifar10.py    # Test CIFAR-10 specifically
python test_hf_dataset.py # Test TinyImageNet
```

## Architecture Overview

### Core Flow-Matching

The model learns a velocity field v_θ(x_t) that transforms noise into data via:
- **Interpolation**: x_t = (1-t) * x_0 + t * x_1
- **Training objective**: v_θ(x_t) → (x_1 - x_0)
- **Sampling**: Euler ODE solver from noise to data

### Model Components

1. **train_flow.py**: Main training loop with flow-matching loss
2. **diffusion_transformer.py**: DiT backbone with AdaLN-Zero conditioning
3. **utils/stable_vae.py**: Optional Stable Diffusion VAE encoder/decoder
4. **utils/train_state.py**: Flax training state management
5. **utils/dataset.py**: HuggingFace dataset streaming implementation

### Model Presets

- **debug**: hidden_size=64, depth=2 (testing only)
- **big**: hidden_size=768, depth=12 (DiT-B equivalent, recommended)
- **semilarge**: hidden_size=1024, depth=22
- **large**: hidden_size=1024, depth=24 (DiT-L)
- **xlarge**: hidden_size=1152, depth=28 (DiT-XL)

### Key Configuration Parameters

- **Model**: `--model.preset`, `--model.patch_size` (2 for VAE, 8 for pixels)
- **Training**: `--lr` (0.0001), `--batch_size`, `--max_steps`
- **Flow**: `--denoise_timesteps`, `--t_sampler` (log-normal/uniform)
- **CFG**: `--class_dropout_prob` (0.1), `--cfg_scale` (inference)
- **VAE**: `--model.use_stable_vae` (0=pixel space, 1=latent space)

## Dataset Support

### HuggingFace (Streaming, Recommended)
- **cifar10**: 32x32, 10 classes
- **cifar100**: 32x32, 100 classes  
- **tiny-imagenet**: 64x64, 200 classes (uses Maysee/tiny-imagenet)

### TensorFlow Datasets
Requires manual compilation via [tfds_builders](https://github.com/kvfrans/tfds_builders):
- **imagenet256**: 256x256 ImageNet
- **celebahq256**: 256x256 CelebA-HQ

## TPU/Multi-Device Training

The codebase uses pmap for multi-device parallelization. Key considerations:
- Batch size must be divisible by device count (8 for v3-8 TPU)
- Use streaming datasets to avoid storage constraints
- Model presets are memory-optimized for different configurations

## Common Development Patterns

### Adding New Datasets
1. Check `utils/dataset.py` for HuggingFace dataset loading pattern
2. Ensure dataset returns dict with 'image' and optionally 'label' keys
3. Images should be normalized to [-1, 1] range

### Modifying Model Architecture
1. Edit `diffusion_transformer.py` for transformer changes
2. Update model presets in `train_flow.py` config
3. Ensure compatibility with AdaLN conditioning

### Debugging Training
1. Use `--model.preset debug` for quick iteration
2. Set `--wandb.mode offline` to disable logging
3. Add `--max_steps 100` for short test runs
4. Check `test_*.py` files for component testing examples

## Important Notes

- Flow-matching uses deterministic sampling (no noise injection during inference)
- Classifier-free guidance significantly improves sample quality (cfg_scale=4 recommended)
- VAE latent space training (patch_size=2) is more efficient than pixel space (patch_size=8)
- The codebase supports both v-prediction and normalized velocity parameterizations