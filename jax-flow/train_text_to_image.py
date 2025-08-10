#!/usr/bin/env python
"""Training script for text-to-image flow matching models."""

try:
    from localutils.debugger import enable_debug
    enable_debug()
except ImportError:
    pass

from typing import Any
import jax.numpy as jnp
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import flax
import optax
import wandb
from ml_collections import config_flags
import ml_collections
import matplotlib.pyplot as plt

from utils.wandb import setup_wandb, default_wandb_config
from utils.train_state import TrainState, target_update
from utils.checkpoint import Checkpoint
from utils.text_image_datasets import create_text_image_dataloader
from diffusion_transformer_text import DiTText

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'conceptual_captions', 'Dataset name.')
flags.DEFINE_string('load_dir', None, 'Load checkpoint from this directory.')
flags.DEFINE_string('save_dir', None, 'Save checkpoints to this directory.')
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('log_interval', 100, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 10000, 'Save interval.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_integer('max_steps', int(100_000), 'Number of training steps.')
flags.DEFINE_integer('image_size', 256, 'Image size.')
flags.DEFINE_integer('use_stable_vae', 0, 'Use Stable Diffusion VAE for latent space.')

flags.DEFINE_integer('debug_overfit', 0, 'Debug overfitting on small dataset.')

model_config = ml_collections.ConfigDict({
    'lr': 0.0001,
    'beta1': 0.9,
    'beta2': 0.99,
    'hidden_size': 768,
    'patch_size': 8,
    'depth': 12,
    'num_heads': 12,
    'mlp_ratio': 4,
    'text_dropout_prob': 0.1,
    'denoise_timesteps': 32,
    'cfg_scale': 7.5,  # Higher CFG scale for text conditioning
    'target_update_rate': 0.9999,
    't_sampler': 'uniform',
    't_conditioning': 1,
    'preset': 'medium',
})

preset_configs = {
    'debug': {
        'hidden_size': 128,
        'patch_size': 16,
        'depth': 2,
        'num_heads': 4,
        'mlp_ratio': 2,
    },
    'small': {
        'hidden_size': 384,
        'patch_size': 8,
        'depth': 6,
        'num_heads': 6,
        'mlp_ratio': 4,
    },
    'medium': {
        'hidden_size': 768,
        'patch_size': 8,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
    },
    'large': {
        'hidden_size': 1024,
        'patch_size': 8,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4,
    },
}

wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'text_to_image_flow',
    'name': 'flow_{dataset_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('model', model_config, lock_config=False)

##############################################
## Flow Matching Functions
##############################################

def get_x_t(images, eps, t):
    """Linear interpolation between noise and data."""
    x_0 = eps  # Noise
    x_1 = images  # Data
    t = jnp.clip(t, 0, 1-0.01)  # Always include a little noise
    return (1-t) * x_0 + t * x_1

def get_v(images, eps):
    """Velocity between noise and data."""
    x_0 = eps
    x_1 = images
    return x_1 - x_0

class TextFlowTrainer(flax.struct.PyTreeNode):
    """Trainer for text-conditioned flow matching."""
    rng: Any
    model: TrainState
    model_eps: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    @partial(jax.pmap, axis_name='data')
    def update(self, images, text_embeddings, pmap_axis='data'):
        new_rng, text_key, time_key, noise_key = jax.random.split(self.rng, 4)

        def loss_fn(params):
            # Sample timestep
            if self.config['t_sampler'] == 'uniform':
                t = jax.random.uniform(time_key, (images.shape[0],), minval=0, maxval=1)
            else:  # log-normal
                t = jax.random.normal(time_key, (images.shape[0],))
                t = 1 / (1 + jnp.exp(-t))

            t_full = t[:, None, None, None]  # [batch, 1, 1, 1]
            eps = jax.random.normal(noise_key, images.shape)
            x_t = get_x_t(images, eps, t_full)
            v_t = get_v(images, eps)

            if self.config['t_conditioning'] == 0:
                t = jnp.zeros_like(t)
            
            # Predict velocity with text conditioning
            v_prime = self.model(
                x_t, t, text_embeddings, 
                train=True, 
                rngs={'text_dropout': text_key}, 
                params=params
            )
            
            loss = jnp.mean((v_prime - v_t) ** 2)
            
            return loss, {
                'l2_loss': loss,
                'v_abs_mean': jnp.abs(v_t).mean(),
                'v_pred_abs_mean': jnp.abs(v_prime).mean(),
            }
        
        grads, info = jax.grad(loss_fn, has_aux=True)(self.model.params)
        grads = jax.lax.pmean(grads, axis_name=pmap_axis)
        info = jax.lax.pmean(info, axis_name=pmap_axis)

        updates, new_opt_state = self.model.tx.update(grads, self.model.opt_state, self.model.params)
        new_params = optax.apply_updates(self.model.params, updates)
        new_model = self.model.replace(step=self.model.step + 1, params=new_params, opt_state=new_opt_state)

        info['grad_norm'] = optax.global_norm(grads)
        info['update_norm'] = optax.global_norm(updates)
        info['param_norm'] = optax.global_norm(new_params)

        # Update EMA model
        new_model_eps = target_update(self.model, self.model_eps, 1-self.config['target_update_rate'])
        if self.config['target_update_rate'] == 1:
            new_model_eps = new_model
        
        new_trainer = self.replace(rng=new_rng, model=new_model, model_eps=new_model_eps)
        return new_trainer, info

    @partial(jax.jit, static_argnames=('cfg'))
    def call_model(self, images, t, text_embeddings, cfg=True, cfg_val=7.5):
        """Call model with optional classifier-free guidance."""
        if self.config['t_conditioning'] == 0:
            t = jnp.zeros_like(t)
        
        if not cfg:
            return self.model_eps(images, t, text_embeddings, train=False, force_drop_ids=False)
        else:
            # Classifier-free guidance: run with and without text
            batch_size = images.shape[0]
            
            # Stack conditional and unconditional
            images_expanded = jnp.tile(images, (2, 1, 1, 1))
            t_expanded = jnp.tile(t, (2,))
            
            # Unconditional uses zeros for text embedding
            text_uncond = jnp.zeros_like(text_embeddings)
            text_full = jnp.concatenate([text_embeddings, text_uncond], axis=0)
            
            # Single forward pass
            v_pred = self.model_eps(images_expanded, t_expanded, text_full, train=False, force_drop_ids=False)
            
            # Split and apply CFG
            v_cond = v_pred[:batch_size]
            v_uncond = v_pred[batch_size:]
            v = v_uncond + cfg_val * (v_cond - v_uncond)
            
            return v

##############################################
## Main Training Loop
##############################################

def main(_):
    # Apply preset config
    preset_dict = preset_configs[FLAGS.model.preset]
    for k, v in preset_dict.items():
        FLAGS.model[k] = v

    np.random.seed(FLAGS.seed)
    print("Using devices", jax.local_devices())
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    print("Device count", device_count)
    print("Global device count", global_device_count)
    
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)
    print("Global Batch:", FLAGS.batch_size)
    print("Node Batch:", local_batch_size)
    print("Device Batch:", local_batch_size // device_count)

    # Setup wandb
    if jax.process_index() == 0:
        setup_wandb(FLAGS.model.to_dict(), **FLAGS.wandb)

    # Create dataset
    print("Loading dataset:", FLAGS.dataset_name)
    
    if FLAGS.dataset_name == "coyo":
        # Use COYO-700M with VAE
        from utils.coyo_dataset import create_coyo_dataloader
        train_loader = create_coyo_dataloader(
            batch_size=local_batch_size,
            image_size=FLAGS.image_size,
            use_vae=FLAGS.use_stable_vae == 1,
            streaming=True,
        )
    elif FLAGS.dataset_name == "mock":
        # Use mock dataset for testing
        from utils.text_image_datasets_mock import create_mock_dataloader
        train_loader = create_mock_dataloader(
            batch_size=local_batch_size,
            image_size=FLAGS.image_size,
            text_embed_dim=768 if FLAGS.use_stable_vae else 384,
        )
    else:
        # Use generic text-image dataset
        from utils.text_image_datasets import create_text_image_dataloader
        train_loader = create_text_image_dataloader(
            dataset_name=FLAGS.dataset_name,
            batch_size=local_batch_size,
            image_size=FLAGS.image_size,
            split="train",
            streaming=True,
            seed=FLAGS.seed
        )
    
    # Get example batch for model initialization
    print("Getting example batch...")
    train_iter = train_loader.create_infinite_iterator()
    example_batch = next(train_iter)
    example_obs = example_batch['image'][:1]
    example_text = example_batch['text_embedding'][:1]
    
    print(f"Image shape: {example_obs.shape}")
    print(f"Text embedding shape: {example_text.shape}")

    # Initialize model
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, param_key, dropout_key = jax.random.split(rng, 3)

    FLAGS.model.image_channels = example_obs.shape[-1]
    FLAGS.model.image_size = example_obs.shape[1]
    
    dit_args = {
        'patch_size': FLAGS.model['patch_size'],
        'hidden_size': FLAGS.model['hidden_size'],
        'depth': FLAGS.model['depth'],
        'num_heads': FLAGS.model['num_heads'],
        'mlp_ratio': FLAGS.model['mlp_ratio'],
        'text_dropout_prob': FLAGS.model['text_dropout_prob'],
        'text_embed_dim': example_text.shape[-1],
    }
    
    model_def = DiTText(**dit_args)
    
    example_t = jnp.zeros((1,))
    model_rngs = {'params': param_key, 'text_dropout': dropout_key}
    params = model_def.init(model_rngs, example_obs, example_t, example_text)['params']
    
    print("Total num of parameters:", sum(x.size for x in jax.tree_util.tree_leaves(params)))
    
    # Create optimizer and training state
    tx = optax.adam(learning_rate=FLAGS.model['lr'], b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'])
    model_ts = TrainState.create(model_def, params, tx=tx)
    model_ts_eps = TrainState.create(model_def, params)
    model = TextFlowTrainer(rng, model_ts, model_ts_eps, FLAGS.model)

    # Load checkpoint if specified
    if FLAGS.load_dir is not None:
        cp = Checkpoint(FLAGS.load_dir)
        model = cp.load_model(model)
        print("Loaded model with step", model.model.step)
        del cp

    # Replicate model across devices
    model = flax.jax_utils.replicate(model, devices=jax.local_devices())
    model = model.replace(rng=jax.random.split(rng, len(jax.local_devices())))

    # Training loop
    pbar = tqdm.tqdm(range(FLAGS.max_steps))
    for i in pbar:
        # Get batch
        batch = next(train_iter)
        batch_images = batch['image'].reshape((device_count, -1, *batch['image'].shape[1:]))
        batch_text = batch['text_embedding'].reshape((device_count, -1, *batch['text_embedding'].shape[1:]))
        
        # Update model
        model, update_info = model.update(batch_images, batch_text)
        
        # Logging
        if i % FLAGS.log_interval == 0:
            update_info = jax.tree.map(lambda x: np.array(x), update_info)
            update_info = jax.tree.map(lambda x: x.mean(), update_info)
            metrics = {f'training/{k}': v for k, v in update_info.items()}
            
            if jax.process_index() == 0:
                wandb.log(metrics, step=i)
                
            pbar.set_description(f"Loss: {update_info['l2_loss']:.4f}")
        
        # Save checkpoint
        if FLAGS.save_dir is not None and i % FLAGS.save_interval == 0 and i > 0:
            if jax.process_index() == 0:
                cp = Checkpoint(FLAGS.save_dir)
                cp.save_model(flax.jax_utils.unreplicate(model))
                print(f"Saved checkpoint at step {i}")
                del cp
    
    print("Training completed!")

if __name__ == '__main__':
    app.run(main)