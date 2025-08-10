# Text-conditioned Diffusion Transformer (DiT-Text)
# Modified from original DiT to support text conditioning instead of class labels

import functools
import math
from typing import Any, Tuple, Optional
import flax.linen as nn
from flax.linen.initializers import xavier_uniform
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections
from einops import rearrange

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union

#################################################################################
#               Embedding Layers for Timesteps and Text                         #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    hidden_size: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        return x

    def timestep_embedding(self, t, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        t = jax.lax.convert_element_type(t, jnp.float32)
        t = t * max_period
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding

class TextEmbedder(nn.Module):
    """
    Projects text embeddings and handles dropout for classifier-free guidance.
    """
    dropout_prob: float
    hidden_size: int
    text_embed_dim: int = 512  # CLIP text embedding dimension

    @nn.compact
    def __call__(self, text_embeddings, train, force_drop_ids=None):
        # Project text embeddings to model dimension
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(text_embeddings)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        
        # Dropout for classifier-free guidance
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            if force_drop_ids is None:
                rng = self.make_rng('text_dropout')
                drop_ids = jax.random.bernoulli(rng, self.dropout_prob, (text_embeddings.shape[0],))
            else:
                drop_ids = force_drop_ids == 1
            
            # Zero out embeddings for dropped samples (unconditional generation)
            x = jnp.where(drop_ids[:, None], jnp.zeros_like(x), x)
        
        return x

class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(inputs)
        x = nn.gelu(x, approximate=False)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)
        return output

class Attention(nn.Module):
    """Multi-head attention."""
    num_heads: int
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    dropout_rate: float = 0.0
    kernel_init: Callable = nn.linear.default_kernel_init
    use_bias: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        batch, seq_len, _ = x.shape
        features = self.qkv_features or x.shape[-1]
        qkv_features = features * 3
        
        qkv = nn.Dense(qkv_features, use_bias=self.use_bias, kernel_init=self.kernel_init, dtype=self.dtype)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        
        attention_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(features // self.num_heads)
        attention_probs = jax.nn.softmax(attention_scores, axis=-1)
        attention_probs = nn.Dropout(rate=self.dropout_rate)(attention_probs, deterministic=deterministic)
        
        out = jnp.einsum('bhqk,bhkd->bhqd', attention_probs, v)
        out = rearrange(out, 'b h l d -> b l (h d)')
        
        out_features = self.out_features or x.shape[-1]
        out = nn.Dense(out_features, use_bias=self.use_bias, kernel_init=self.kernel_init, dtype=self.dtype)(out)
        return out

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, c, train=True):
        # c is conditioning vector from timestep + text
        
        gate_msa, gate_mlp = jnp.split(
            nn.Dense(self.hidden_size * 6, use_bias=False)(nn.silu(c)), 
            [3 * self.hidden_size], 
            axis=-1
        )
        shift_msa, scale_msa, gate_msa = jnp.split(gate_msa, 3, axis=-1)
        shift_mlp, scale_mlp, gate_mlp = jnp.split(gate_mlp, 3, axis=-1)
        
        # Attention block
        norm_x = nn.LayerNorm()(x)
        norm_x = modulate(norm_x, shift_msa[:, None, :], scale_msa[:, None, :])
        attn_out = Attention(num_heads=self.num_heads)(norm_x, deterministic=not train)
        x = x + gate_msa[:, None, :] * attn_out
        
        # MLP block
        norm_x = nn.LayerNorm()(x)
        norm_x = modulate(norm_x, shift_mlp[:, None, :], scale_mlp[:, None, :])
        mlp_out = MlpBlock(
            mlp_dim=int(self.hidden_size * self.mlp_ratio),
            dropout_rate=0.0
        )(norm_x, deterministic=not train)
        x = x + gate_mlp[:, None, :] * mlp_out
        
        return x

class FinalLayer(nn.Module):
    """The final layer of DiT."""
    hidden_size: int
    patch_size: int
    out_channels: int

    @nn.compact
    def __call__(self, x, c):
        c = nn.silu(c)
        shift, scale = jnp.split(nn.Dense(self.hidden_size * 2, use_bias=False)(c), 2, axis=-1)
        x = modulate(nn.LayerNorm()(x), shift[:, None, :], scale[:, None, :])
        x = nn.Dense(self.patch_size * self.patch_size * self.out_channels, use_bias=False)(x)
        return x

class DiTText(nn.Module):
    """
    Text-conditioned Diffusion Transformer.
    """
    patch_size: int = 2
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    text_dropout_prob: float = 0.1
    text_embed_dim: int = 512  # CLIP text embedding dimension

    def patchify(self, x):
        B, H, W, C = x.shape
        pH = H // self.patch_size
        pW = W // self.patch_size
        x = x.reshape((B, pH, self.patch_size, pW, self.patch_size, C))
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = x.reshape((B, pH * pW, self.patch_size * self.patch_size * C))
        return x

    def unpatchify(self, x, H, W, C):
        pH = H // self.patch_size
        pW = W // self.patch_size
        x = x.reshape((x.shape[0], pH, pW, self.patch_size, self.patch_size, C))
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        x = x.reshape((x.shape[0], H, W, C))
        return x

    @nn.compact
    def __call__(self, x, t, text_embeddings, train=True, force_drop_ids=None, rngs=None):
        B, H, W, C = x.shape
        
        # Patchify input
        x = self.patchify(x)
        num_patches = x.shape[1]
        
        # Patch embedding
        x = nn.Dense(self.hidden_size, use_bias=False)(x)
        
        # Add positional embedding
        pos_embed = self.param(
            'pos_embed',
            nn.initializers.normal(0.02),
            (1, num_patches, self.hidden_size)
        )
        x = x + pos_embed
        
        # Timestep embedding
        t_emb = TimestepEmbedder(self.hidden_size)(t)
        
        # Text embedding with optional dropout
        text_emb = TextEmbedder(
            dropout_prob=self.text_dropout_prob,
            hidden_size=self.hidden_size,
            text_embed_dim=self.text_embed_dim
        )(text_embeddings, train, force_drop_ids)
        
        # Combine timestep and text embeddings
        c = t_emb + text_emb
        
        # DiT blocks
        for i in range(self.depth):
            x = DiTBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                name=f'DiTBlock_{i}'
            )(x, c, train)
        
        # Final layer
        x = FinalLayer(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=C
        )(x, c)
        
        # Unpatchify
        x = self.unpatchify(x, H, W, C)
        
        return x