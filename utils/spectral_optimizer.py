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
    fan_in: int = flax.struct.field(pytree_node=False)
    def unbox(self):
        return self.value
    def replace_boxed(self, value):
        return self.replace(value=value)
    def add_axis(self, index, params):
        return self
    def remove_axis(self, index, params):
        return self

def spectral_init(global_scale=1, in_axis=-2, out_axis=-1, batch_axis=(), dtype=jnp.float_):
    def init(key, shape: core.Shape, dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        named_shape = core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(named_shape, in_axis, out_axis, batch_axis)
        scale = global_scale * jnp.sqrt(fan_out / fan_in) * (1 / (jnp.sqrt(fan_in) + jnp.sqrt(fan_out)))
        scale = scale / jnp.array(.87962566103423978, dtype) # Scale by truncated normal constant.
        param = jax.random.truncated_normal(key, -2, 2, shape, dtype) * scale
        param = SpectralNormalizedParameter(value=param, fan_in=fan_in)
        return param
    return init

def scale_spectral_norm():
    def init_fn(params):
        return optax.EmptyState()
    def update_fn(updates, state, params=None):
        del params
        def scale_updates(update): # update is either a jax array or a MaximalUpdateParametrizationMetadata.
            if isinstance(update, SpectralNormalizedParameter):
                return nn.meta.replace_boxed(update, nn.meta.unbox(update) / update.fan_in)
            return update
        updates = jax.tree_util.tree_map(scale_updates, updates, is_leaf=lambda leaf: isinstance(leaf, SpectralNormalizedParameter))
        return updates, state
    return optax.GradientTransformation(init_fn, update_fn)