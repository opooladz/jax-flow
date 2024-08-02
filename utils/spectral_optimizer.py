import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import chex
import flax

from jax._src import core
from jax._src import dtypes
from jax._src.nn.initializers import _compute_fans

class SpectralNormalizedParameter(flax.struct.PyTreeNode, nn.meta.AxisMetadata):
    value: chex.Array
    lr_scale: int = flax.struct.field(pytree_node=False)
    def unbox(self):
        return self.value
    def replace_boxed(self, value):
        return self.replace(value=value)
    def add_axis(self, index, params):
        return self
    def remove_axis(self, index, params):
        return self

def spectral_init(init_scale=1, lr_scale=1, in_axis=-2, out_axis=-1, batch_axis=(), dtype=jnp.float_):
    def init(key, shape: core.Shape, dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        named_shape = core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(named_shape, in_axis, out_axis, batch_axis)
        scale = init_scale * jnp.sqrt(fan_out / fan_in) * (1 / (jnp.sqrt(fan_in) + jnp.sqrt(fan_out)))
        scale = scale / jnp.array(.87962566103423978, dtype) # Scale by truncated normal constant.
        param = jax.random.truncated_normal(key, -2, 2, shape, dtype) * scale
        true_lr_scale = lr_scale / fan_in
        param = SpectralNormalizedParameter(value=param, lr_scale=true_lr_scale)
        return param
    return init

def scale_spectral_norm():
    def init_fn(params):
        return optax.EmptyState()
    def update_fn(updates, state, params=None):
        del params
        def scale_updates(update): # update is either a jax array or a SpectralNormalizedParameter.
            if isinstance(update, SpectralNormalizedParameter):
                return nn.meta.replace_boxed(update, nn.meta.unbox(update) * update.lr_scale)
            return update
        updates = jax.tree_util.tree_map(scale_updates, updates, is_leaf=lambda leaf: isinstance(leaf, SpectralNormalizedParameter))
        return updates, state
    return optax.GradientTransformation(init_fn, update_fn)


# def scale_layers(mult_patch, mult_embed, mult_final):
#     def init_fn(params):
#         return optax.EmptyState()
#     def update_fn(updates, state, params=None):
#         del params
#         def scale_updates(path, update):
#             print(path)
#             breakpoint()
#             if isinstance(update, SpectralNormalizedParameter):
#                 if 'PatchEmbed' in path:
#                     return nn.meta.replace_boxed(update, nn.meta.unbox(update) * mult_patch)
#                 if 'LabelEmbedder' in path:
#                     return nn.meta.replace_boxed(update, nn.meta.unbox(update) * mult_embed)
#                 if 'FinalLayer' in path:
#                     return nn.meta.replace_boxed(update, nn.meta.unbox(update) * mult_final)
#             return update
#         updates = jax.tree_util.tree_map_with_path(scale_updates, updates, is_leaf=lambda leaf: isinstance(leaf, SpectralNormalizedParameter))
#         return updates, state
#     return optax.GradientTransformation(init_fn, update_fn)