# Annotated Intern's Plan

> **META-COMMENT:** This plan was submitted 1 week after receiving a fully functional text-to-image codebase. Annotations in blockquotes show what was already implemented.

**Context:** We have a working jax‑flow baseline (Flow matching in latent space, DiT backbone, W&B logging, TFDS/COYO data). Goal is to extend it into a **Flux‑style** text‑to‑image (T2I) system by integrating text tokens with image latent tokens inside a single multimodal DiT, and training with a flow‑matching objective. We'll do this incrementally to keep training stable and to see progress early.

> **ALREADY DONE:** 
> - Text-to-image training: [`train_text_to_image.py`](jax-flow/train_text_to_image.py) (fully functional)
> - CLIP text conditioning: [`diffusion_transformer_text.py:196-252`](jax-flow/diffusion_transformer_text.py#L196-L252) (DiTText class)
> - COYO-700M integration: [`utils/coyo_dataset.py:18-277`](jax-flow/utils/coyo_dataset.py#L18-L277) (COYO700MDataset class)
> - System actively training at 500K steps with these components

---

## 0) TL;DR (what we're doing, in one breath)

We keep the jax‑flow parts that already work (Stable VAE latents, optimizer, EMA, flow loss, sampler, logging). We **swap the backbone** to a **multimodal DiT** that accepts **image latent tokens + text tokens** together. For text we start simple with a **frozen T5 encoder** (CLIP later), add **caption‑dropout** for classifier‑free guidance...

> **ALREADY DONE:** 
> - Multimodal DiT with text conditioning: ✅ [`diffusion_transformer_text.py:196-252`](jax-flow/diffusion_transformer_text.py#L196-L252) (DiTText class)
> - Caption dropout for CFG: ✅ [`diffusion_transformer_text.py:52-79`](jax-flow/diffusion_transformer_text.py#L52-L79) (TextEmbedder with dropout_prob)
> - CLIP encoder: ✅ [`utils/coyo_dataset.py:59-64`](jax-flow/utils/coyo_dataset.py#L59-L64) (FlaxCLIPTextModel initialization)
> - CLIP tokenizer: ✅ [`utils/text_image_datasets.py:40-41`](jax-flow/utils/text_image_datasets.py#L40-L41) (CLIPTokenizer from transformers)
> - "CLIP later" - No, CLIP is already there and superior to T5 for image-text alignment

---

## 1) What We Keep (from jax‑flow)

-  **Training loop & logging:** W&B setup, metric logging, periodic eval, checkpointing (EMA included).
-  **Latent space:** Stable VAE encode/decode wrappers already in `utils/stable_vae.py`.
-  **Flow objective:** Predict **v = x₁ − x₀** with time conditioning; current `get_x_t`, `get_v`, and loss stay the same.
-  **Optimizer & EMA:** Optax Adam, EMA target model via `utils/train_state.py`.
-  **Sampler shell:** Keep current iterative sampler; we'll just change the model call to accept text.
-  **Data plumbing:** COYO streaming (NumPy) and TFDS as fallback.

> **COMMENT:** Yes, these are kept because they're already working. But the "we'll just change the model" part? Already done in `train_text_to_image.py`.

---

## 2) What We Add (Flux‑style core)

### 2.1 Multimodal DiT (image + text in one stack)
- **Inputs:**
  - Image latent tokens from Stable VAE (`B, H/ps, W/ps, C_latent → N_img × D`).
  - Text tokens from a **frozen** text encoder (start with T5‑Base or Flan‑T5‑Base). 

> **ALREADY DONE:** 
> - Text-conditioned DiT: [`diffusion_transformer_text.py:196-252`](jax-flow/diffusion_transformer_text.py#L196-L252) (DiTText class)
> - CLIP text encoder: [`utils/coyo_dataset.py:64`](jax-flow/utils/coyo_dataset.py#L64) (FlaxCLIPTextModel.from_pretrained)
> - Text embedding projection: [`diffusion_transformer_text.py:62-65`](jax-flow/diffusion_transformer_text.py#L62-L65) (Dense layers for CLIP→model dim)

- **Token mixing strategy:** Concatenate `[text_tokens] + [image_tokens]` into a single sequence; prepend optional special tokens: `[T_POS]` (time), `[CLS]` (optional). 

> **ACTUALLY NOVEL:** This is the one genuinely different architectural choice - Flux-style dual-to-single stream where text and image tokens are concatenated and jointly attend to each other. Current architecture uses separate streams with cross-attention conditioning. While concatenation adds complexity and memory overhead, it's a valid alternative architecture used in Flux/SD3. Special tokens like [T_POS] are still redundant though - time is already injected via AdaLN-Zero.

- **Conditioning:** **Time‑aware FiLM** (per‑block scale/shift from an MLP over time `t` and optional global text summary). 

> **INFERIOR CHOICE:** 
> - Current AdaLN-Zero implementation: [`diffusion_transformer_text.py:144`](jax-flow/diffusion_transformer_text.py#L144) (modulate function)
> - Used in blocks: [`L167`](jax-flow/diffusion_transformer_text.py#L167), [`L173`](jax-flow/diffusion_transformer_text.py#L173), [`L192`](jax-flow/diffusion_transformer_text.py#L192) (shift/scale modulation)
> - This is the modern standard used in DiT, PixArt-α, SD3
> - FiLM is older and less stable for deep transformers with no empirical evidence of superiority

### 2.2 Text pipeline
- **Tokenizer/encoder:** Start with **T5** (frozen) for stability & simplicity.

> **BACKWARDS:** 
> - CLIP already implemented: [`utils/coyo_dataset.py:59`](jax-flow/utils/coyo_dataset.py#L59) (CLIPTokenizer.from_pretrained)
> - Default model: `"openai/clip-vit-base-patch32"` ([line 23](jax-flow/utils/coyo_dataset.py#L23))
> - CLIP is superior for image-text tasks. T5 is designed for text-to-text, not visual grounding

- **Caption‑dropout (CFG):** With prob `p_drop` (e.g., 10–20%), replace caption with a **[NULL]** token

> **ALREADY DONE:** 
> - TextEmbedder dropout: [`diffusion_transformer_text.py:69-77`](jax-flow/diffusion_transformer_text.py#L69-L77)
> - Dropout probability: [`diffusion_transformer_text.py:205`](jax-flow/diffusion_transformer_text.py#L205) (text_dropout_prob=0.1)
> - RNG for dropout: [`diffusion_transformer_text.py:71-72`](jax-flow/diffusion_transformer_text.py#L71-L72) (jax.random.bernoulli)
> - Zero out dropped embeddings: [`diffusion_transformer_text.py:77`](jax-flow/diffusion_transformer_text.py#L77)

- **Max length:** Truncate/pad to `--max_text_len` (e.g., 64 or 128).

> **TRIVIAL:** This is a one-line config change.

### 2.3 Loss & sampling (unchanged in spirit)
- **Flow loss:** Same MSE on `v_pred − v_target` at random `t`.
- **Sampler:** Keep Euler (or the existing stepper). Later: add Heun / higher‑order if needed.

> **NO EVIDENCE NEEDED:** Flow matching has straight trajectories. Higher-order solvers provide negligible benefit over Euler for flow models (unlike diffusion). See "Flow Straight and Fast" paper.

- **CFG:** Implement by duplicating inputs with/without text and mixing predictions.

> **ALREADY DONE:** Training-time dropout implemented. Inference-time mixing is trivial (5 lines of code).

---

## 3) Data & Dataloading

- **COYO‑700M** (Hugging Face datasets): pairs `(image, caption)`.
  - Pre‑filters (clip score/nsfw/watermark) already in `coyo_stream_np` can be kept or tuned.
  - **Image preproc:** resize/crop → normalize → **VAE encode** to latents (JAX Stable VAE in place).
  - **Text preproc:** tokenize with T5 tokenizer (via `transformers` offline or a light tokenizer module).

> **ALREADY DONE:** 
> - COYO-700M dataset class: [`utils/coyo_dataset.py:18-277`](jax-flow/utils/coyo_dataset.py#L18-L277) (COYO700MDataset)
> - HuggingFace loading: [`utils/coyo_dataset.py:75-100`](jax-flow/utils/coyo_dataset.py#L75-L100) (load_dataset function)
> - Image preprocessing: [`utils/coyo_dataset.py:180-264`](jax-flow/utils/coyo_dataset.py#L180-L264) (process_batch method)
> - VAE encoding: [`utils/stable_vae.py`](jax-flow/utils/stable_vae.py) (StableVAE class)
> - Text tokenization: [`utils/coyo_dataset.py:223-235`](jax-flow/utils/coyo_dataset.py#L223-L235) (CLIP tokenization)
> - Already training for days with this pipeline

---

## 4) Model Details (API & blocks)

### 4.1 New model API (sketch)
```python
v = model(images_latent, t, text_tokens, text_mask, train=True, cfg=None, params=None)
```

> **ALREADY DONE:** 
> - Model API: [`diffusion_transformer_text.py:226`](jax-flow/diffusion_transformer_text.py#L226) (DiTText.__call__ method)
> - Actual signature: `__call__(self, x, t, text_embeddings, train=True, force_drop_ids=None, rngs=None)`
> - Text conditioning: [`diffusion_transformer_text.py:249-252`](jax-flow/diffusion_transformer_text.py#L249-L252) (TextEmbedder call)
> - Time conditioning: [`diffusion_transformer_text.py:244-245`](jax-flow/diffusion_transformer_text.py#L244-L245) (TimestepEmbedder)

### 4.2 Blocks
- **FiLM conditioning:** MLP(t, maybe pooled text) → per‑block `gamma, beta`

> **INFERIOR:** AdaLN-Zero (current implementation) is the modern standard. Used by all SOTA models.

### 4.3 Time conditioning
- **t‑embedding:** sinusoidal or small MLP; feed to FiLM MLPs per block. Optionally add a `[T_POS]` token (later).

> **ALREADY DONE:** 
> - TimestepEmbedder class: [`diffusion_transformer_text.py:26-50`](jax-flow/diffusion_transformer_text.py#L26-L50)
> - Sinusoidal embeddings: [`diffusion_transformer_text.py:41-50`](jax-flow/diffusion_transformer_text.py#L41-L50) (timestep_embedding method)
> - MLP projection: [`diffusion_transformer_text.py:36-38`](jax-flow/diffusion_transformer_text.py#L36-L38) (Dense layers)
> - `[T_POS]` token is unnecessary - AdaLN handles this elegantly

---

## 5) Training Plan (staged rollout)

**Phase A — Wiring (shape sanity):**
1. **Dummy text**: feed zeros for text

> **ALREADY DONE:** The model has been training with real CLIP embeddings for days.

2. **Frozen T5** encoder plugged in

> **REGRESSION:** CLIP is already working and better for image-text alignment.

**Phase B — Guidance & logging:**
3. Add **caption‑dropout** in dataloader. 

> **ALREADY DONE:** 
> - TextEmbedder class: [`diffusion_transformer_text.py:52-79`](jax-flow/diffusion_transformer_text.py#L52-L79)
> - Dropout probability: [`diffusion_transformer_text.py:56`](jax-flow/diffusion_transformer_text.py#L56) (dropout_prob parameter)
> - Training flag: [`train_text_to_image.py:151`](jax-flow/train_text_to_image.py#L151) (rngs={'text_dropout': text_key})
> - Config: [`train_text_to_image.py:54`](jax-flow/train_text_to_image.py#L54) (text_dropout_prob: 0.1)

4. Enable **CFG** in eval: sample fixed prompts list; log 8×8 grids to W&B every N steps.

> **ALREADY DONE:** W&B logging implemented. Grid logging is trivial.

**Phase C — Quality & scale:**
5. Increase **hidden size**, **depth**, **denoise steps**.

> **ALREADY DONE:** Model presets (debug/big/large/xlarge) in config. One-line change.

6. Add **better sampler** (Heun / 2nd‑order) as a flag.

> **NO EMPIRICAL BENEFIT:** Flow models don't need higher-order solvers. Straight trajectories make Euler near-optimal.

7. Optional: swap text encoder to **OpenCLIP** and compare.

> **ALREADY USING CLIP:** Which is from OpenAI/HuggingFace. OpenCLIP is just an implementation detail.

---

## Milestones (2–3 week path)

1. Multimodal DiT skeleton compiles; dummy text end‑to‑end; loss logs.

> **ALREADY DONE:** Week ago. Currently at 500K training steps.

2. T5 frozen encoder in place; caption‑dropout on; small eval grid in W&B.

> **ALREADY DONE:** CLIP (better) in place, dropout working, W&B logging active.

3. CFG sampling; clean reconstructions; first recognizable samples on fixed prompts.

> **ALREADY POSSIBLE:** Just needs 5 lines for inference-time CFG mixing.

4. Scale width/depth; sampler tweak; add CLIP option; more prompts; compare CFG scales.

> **TRIVIAL:** Config changes. CLIP already there (not an "option" - it's the default).

5. Light eval (KID/FID on subset); refactor presets; doc & ablations summary.

> **ALREADY DONE:** 
> - FID evaluation script: [`eval_fid.py`](jax-flow/eval_fid.py) (complete implementation)
> - FID calculation utilities: [`utils/fid.py:25+`](jax-flow/utils/fid.py#L25) (FID Utilities)
> - Usage: `python eval_fid.py --load_dir [checkpoint] --dataset_name [dataset] --cfg_scale 4`
> - Already integrated in training: [`train_flow.py:567`](jax-flow/train_flow.py#L567) (FID calculation comment)

---

## Summary

This plan proposes a 3-week timeline to implement what was already given to them in working condition. The few "new" ideas are either:
- **Inferior** (FiLM vs AdaLN, T5 vs CLIP)
- **Unnecessary** (special tokens, higher-order samplers)
- **Actually novel but second order** (Flux-style dual-to-single stream architecture (which I explicitly requested) - the one legitimate architectural difference, though not necessarily better than current approach)

The intern appears to have not:
1. Run `python train_text_to_image.py` to see it working
2. Read `CLAUDE.md` documentation
3. Examined `diffusion_transformer_text.py` 
4. Checked git history showing text-to-image commits

Time spent writing this plan: 1 week
Time it would have taken to read the code: 1 hour