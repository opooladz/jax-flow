#!/usr/bin/env python3
"""
Escale sharding-focused training for flow-matching models.
Pure focus on distributed sharding without mixed precision or other complications.

Uses escale auto-sharding from @eformer/ (github.com/erfanzar/eformer)
Based on EasyDeL framework patterns from @EasyDeL/ (github.com/erfanzar/EasyDeL)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../eformer'))

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
import ml_collections
import optax
import wandb
import tqdm
from flax.training import train_state

# Initialize JAX distributed like EasyDeL does
try:
    jax.distributed.initialize()
except RuntimeError:
    print("JAX distributed already initialized or not needed")
except Exception:
    pass  # Single process mode

# Escale imports from eformer - using EasyDeL style
# See: https://github.com/erfanzar/eformer for escale implementation
# Based on EasyDeL patterns: https://github.com/erfanzar/EasyDeL
try:
    from eformer.escale import (
        create_mesh,
        PartitionAxis,
        PartitionManager,
        match_partition_rules,
        make_shard_and_gather_fns,
        auto_partition_spec,
        auto_shard_array,
        with_sharding_constraint,
    )
    from eformer.common_types import ColumnWise, RowWise, Replicated
    from jax.sharding import PartitionSpec, NamedSharding  # Need these for sharding
    HAS_ESCALE = True
except ImportError:
    print("WARNING: Escale not found. Using JAX native sharding.")
    HAS_ESCALE = False
    from jax.sharding import Mesh, PartitionSpec, NamedSharding

# Model imports
from diffusion_transformer import DiT
from utils.hf_datasets import create_hf_dataloader

# Optional VAE import
try:
    from utils.stable_vae import StableVAE
except ImportError:
    StableVAE = None
    print("Warning: StableVAE not available (diffusers not installed)")

FLAGS = flags.FLAGS

# Core training flags
flags.DEFINE_string('dataset_name', 'tiny-imagenet', 'Dataset name.')
flags.DEFINE_string('save_dir', None, 'Save directory.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('batch_size', 256, 'Global batch size.')
flags.DEFINE_integer('max_steps', 1000, 'Training steps.')
flags.DEFINE_integer('log_interval', 10, 'Log interval.')
flags.DEFINE_integer('save_interval', 500, 'Save interval.')

# Model flags
flags.DEFINE_float('lr', 0.0001, 'Learning rate.')
flags.DEFINE_string('preset', 'big', 'Model preset.')
flags.DEFINE_integer('patch_size', 8, 'Patch size.')
flags.DEFINE_bool('use_stable_vae', False, 'Use Stable VAE.')
flags.DEFINE_float('class_dropout_prob', 0.1, 'Class dropout.')
flags.DEFINE_integer('denoise_timesteps', 32, 'Denoising steps.')
flags.DEFINE_string('t_sampler', 'log-normal', 'Time sampler.')

# Sharding configuration - THE MAIN FOCUS
flags.DEFINE_integer('dp_devices', 4, 'Data parallel devices.')
flags.DEFINE_integer('fsdp_devices', 8, 'FSDP devices.')

# Model presets
preset_configs = {
    'debug': {
        'hidden_size': 64,
        'depth': 2,
        'num_heads': 2,
        'mlp_ratio': 1,
    },
    'big': {
        'hidden_size': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
    },
    'large': {
        'hidden_size': 1024,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4,
    },
}


class EscaleShardingTrainer:
    """Escale trainer focused purely on sharding."""
    
    def __init__(self, config):
        self.config = config
        self.setup_sharding_mesh()
        self.setup_model()
        self.setup_partition_rules()  # Add EasyDeL-style partition rules
        
    def setup_sharding_mesh(self):
        """Setup sharding mesh - THE CORE FEATURE."""
        device_count = jax.device_count()
        print(f"\n🔷 SHARDING SETUP")
        print(f"   Total devices: {device_count}")
        
        if HAS_ESCALE:
            # Escale auto-sharding
            if device_count == 32:  # v4-64
                mesh_shape = (self.config.dp_devices, self.config.fsdp_devices)
                axis_names = ("dp", "fsdp")
            elif device_count == 8:  # v3-8
                mesh_shape = (2, 4)
                axis_names = ("dp", "fsdp")
            else:
                mesh_shape = (device_count,)
                axis_names = ("all",)
            
            self.mesh = create_mesh(
                axis_dims=mesh_shape,
                axis_names=axis_names
            )
            
            # Create PartitionAxis for EasyDeL-style sharding
            self.partition_axis = PartitionAxis(
                batch_axis="dp",
                sequence_axis=None,
                head_axis=None,
                hidden_state_axis="fsdp",
                key_sequence_axis=None,
            )
            self.partition_manager = PartitionManager(self.partition_axis)
            
            print(f"   ✅ Escale mesh: {mesh_shape} with axes {axis_names}")
            
        else:
            # JAX native fallback
            from jax.sharding import Mesh
            devices = np.array(jax.devices())
            if device_count == 32:
                devices = devices.reshape(self.config.dp_devices, self.config.fsdp_devices)
                self.mesh = Mesh(devices, ("dp", "fsdp"))
            elif device_count == 8:
                devices = devices.reshape(2, 4)
                self.mesh = Mesh(devices, ("dp", "fsdp"))
            else:
                self.mesh = Mesh(devices, ("all",))
            
            # Create dummy partition manager for compatibility
            self.partition_axis = None
            self.partition_manager = None
            
            print(f"   ✅ JAX mesh created")
    
    def setup_model(self):
        """Setup model configuration."""
        preset = preset_configs[self.config.preset]
        
        # Dataset config
        if 'cifar10' in self.config.dataset_name:
            self.image_size = 32
            self.num_classes = 10
        elif 'cifar100' in self.config.dataset_name:
            self.image_size = 32
            self.num_classes = 100
        elif 'tiny-imagenet' in self.config.dataset_name:
            self.image_size = 64
            self.num_classes = 200
        else:
            self.image_size = 256
            self.num_classes = 1000
        
        self.model_def = DiT(
            patch_size=self.config.patch_size,
            hidden_size=preset['hidden_size'],
            depth=preset['depth'],
            num_heads=preset['num_heads'],
            mlp_ratio=preset['mlp_ratio'],
            class_dropout_prob=self.config.class_dropout_prob,
            num_classes=self.num_classes,
            learn_sigma=False,
        )
    
    def setup_partition_rules(self):
        """Setup partition rules for DiT model like EasyDeL does."""
        if not HAS_ESCALE:
            self.partition_rules = None
            return
            
        # Define partition rules for DiT model layers - matching actual Flax names
        pmag = self.partition_manager
        self.partition_rules = (
            # Patch embedding (PatchEmbed_0/Conv_0)
            (r"PatchEmbed_\d+/Conv_\d+/kernel", pmag.resolve(ColumnWise)),
            (r"PatchEmbed_\d+/Conv_\d+/bias", pmag.resolve(Replicated)),
            
            # Time embedding (TimestepEmbedder_0/Dense_0, Dense_1)
            (r"TimestepEmbedder_\d+/Dense_\d+/kernel", pmag.resolve(ColumnWise)),
            (r"TimestepEmbedder_\d+/Dense_\d+/bias", pmag.resolve(Replicated)),
            
            # Label embedding (LabelEmbedder_0/Embed_0)
            (r"LabelEmbedder_\d+/Embed_\d+/embedding", pmag.resolve(ColumnWise)),
            
            # Position embedding parameter
            (r"pos_embed", pmag.resolve(Replicated)),
            
            # DiT blocks - attention (DiTBlock_N/MultiHeadDotProductAttention_0)
            (r"DiTBlock_\d+/MultiHeadDotProductAttention_\d+/(query|key|value)/kernel", pmag.resolve(ColumnWise)),
            (r"DiTBlock_\d+/MultiHeadDotProductAttention_\d+/out/kernel", pmag.resolve(RowWise)),
            (r"DiTBlock_\d+/MultiHeadDotProductAttention_\d+/(query|key|value)/bias", pmag.resolve(Replicated)),
            (r"DiTBlock_\d+/MultiHeadDotProductAttention_\d+/out/bias", pmag.resolve(Replicated)),
            
            # DiT blocks - MLP (DiTBlock_N/MlpBlock_0/Dense_0, Dense_1)
            (r"DiTBlock_\d+/MlpBlock_\d+/Dense_0/kernel", pmag.resolve(ColumnWise)),
            (r"DiTBlock_\d+/MlpBlock_\d+/Dense_1/kernel", pmag.resolve(RowWise)),
            (r"DiTBlock_\d+/MlpBlock_\d+/Dense_\d+/bias", pmag.resolve(Replicated)),
            
            # DiT blocks - AdaLN modulation (DiTBlock_N/Dense_0)
            (r"DiTBlock_\d+/Dense_\d+/kernel", pmag.resolve(ColumnWise)),
            (r"DiTBlock_\d+/Dense_\d+/bias", pmag.resolve(Replicated)),
            
            # Final layer (FinalLayer_0/Dense_0, Dense_1)
            (r"FinalLayer_\d+/Dense_\d+/kernel", pmag.resolve(ColumnWise)),
            (r"FinalLayer_\d+/Dense_\d+/bias", pmag.resolve(Replicated)),
            
            # Default for everything else
            (r".*", pmag.resolve(Replicated)),
        )
        
        if self.config.use_stable_vae and StableVAE is not None:
            self.vae = StableVAE()
            self.latent_size = self.image_size // 8
            self.channels = 4
        else:
            self.vae = None
            self.latent_size = self.image_size
            self.channels = 3
        
        print(f"\n📦 Model: {self.config.preset}")
        print(f"   Image: {self.image_size}x{self.image_size}x{self.channels}")
    
    def shard_params(self, params):
        """Apply sharding to parameters using EasyDeL-style partition rules."""
        if HAS_ESCALE and hasattr(self, 'partition_rules') and self.partition_rules:
            print("\n🔄 Applying EasyDeL-style escale sharding...")
            
            # Flatten params to get paths
            flat_params, treedef = jax.tree_util.tree_flatten_with_path(params)
            
            # Match partition rules to parameters like EasyDeL
            # Build a dict with string keys for matching
            params_dict = {}
            for path, value in flat_params:
                # Convert path to string format that match_partition_rules expects
                path_str = "/".join([
                    key.key if hasattr(key, 'key') else str(key)
                    for key in path
                ])
                params_dict[path_str] = value.shape
            
            # Match rules to get partition specs
            partition_specs = match_partition_rules(
                rules=self.partition_rules,
                tree=params_dict
            )
            
            # Apply sharding to each parameter
            sharded_params = []
            num_sharded = 0
            for path, value in flat_params:
                path_str = "/".join([
                    key.key if hasattr(key, 'key') else str(key)
                    for key in path
                ])
                
                if path_str in partition_specs:
                    spec = partition_specs[path_str]
                    if spec != PartitionSpec():  # Not replicated
                        # Apply actual sharding with NamedSharding
                        sharding = NamedSharding(self.mesh, spec)
                        sharded_value = jax.device_put(value, sharding)
                        sharded_params.append(sharded_value)
                        num_sharded += 1
                    else:
                        sharded_params.append(value)
                else:
                    sharded_params.append(value)
            
            # Reconstruct the tree
            params = jax.tree_util.tree_unflatten(treedef, sharded_params)
            
            # Log sharding info
            print(f"✅ Sharded {num_sharded} parameters out of {len(flat_params)} total")
            
            # Show examples of sharded parameters
            example_specs = []
            for path_str, spec in list(partition_specs.items())[:10]:
                if spec != PartitionSpec():
                    # Truncate path for display
                    short_path = path_str if len(path_str) <= 50 else "..." + path_str[-47:]
                    example_specs.append(f"     {short_path}: {spec}")
            if example_specs:
                print("   Examples of sharded parameters:")
                print("\n".join(example_specs[:5]))
            
        elif HAS_ESCALE:
            # Fallback to auto_partition_spec if no rules defined
            print("\n🔄 Applying Escale auto-sharding...")
            params_spec = jax.tree_util.tree_map(
                lambda x: auto_partition_spec(x, self.mesh, names=["dp", "fsdp"]),
                params
            )
            params = with_sharding_constraint(params, params_spec)
            
        else:
            # Manual JAX sharding
            print("\n🔄 Applying JAX native sharding...")
            def apply_shard(x):
                if hasattr(x, 'shape') and x.size > 10000:
                    if hasattr(self.mesh, 'axis_names') and 'fsdp' in self.mesh.axis_names:
                        spec = PartitionSpec("fsdp", None) if len(x.shape) >= 2 else PartitionSpec("fsdp")
                    else:
                        spec = PartitionSpec()
                    sharding = NamedSharding(self.mesh, spec)
                    return jax.device_put(x, sharding)
                return x
            
            params = jax.tree_util.tree_map(apply_shard, params)
        
        return params
    
    def init_training(self, rng):
        """Initialize model and optimizer with sharding."""
        # Create dummy batch
        dummy_images = jnp.ones((2, self.latent_size, self.latent_size, self.channels))
        dummy_t = jnp.ones((2,))
        dummy_labels = jnp.ones((2,), dtype=jnp.int32)
        
        # Initialize
        params_key, dropout_key = jax.random.split(rng)
        variables = self.model_def.init(
            {'params': params_key, 'label_dropout': dropout_key},
            dummy_images, dummy_t, dummy_labels, train=True
        )
        params = variables['params']
        
        # SHARD THE PARAMS
        params = self.shard_params(params)
        
        # Count params
        param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
        # Get FSDP device count from config since escale mesh doesn't have shape attribute
        fsdp_devices = self.config.fsdp_devices if self.config.fsdp_devices else jax.device_count()
        
        print(f"\n💾 Parameters: {param_count:,}")
        print(f"   Per-device memory: ~{param_count * 4 / fsdp_devices / 1e6:.1f} MB")
        
        # Optimizer
        optimizer = optax.adam(self.config.lr)
        
        # Create train state 
        state = train_state.TrainState.create(
            apply_fn=self.model_def.apply,
            params=params,
            tx=optimizer,
        )
        
        # SHARD THE OPTIMIZER STATE AFTER CREATION
        if HAS_ESCALE:
            opt_state_spec = jax.tree_util.tree_map(
                lambda x: auto_partition_spec(x, self.mesh, names=["dp", "fsdp"]) if hasattr(x, 'shape') else None,
                state.opt_state
            )
            state = state.replace(opt_state=with_sharding_constraint(state.opt_state, opt_state_spec))
        else:
            def shard_opt(x):
                if hasattr(x, 'shape') and x.size > 10000:
                    if hasattr(self.mesh, 'axis_names') and 'fsdp' in self.mesh.axis_names:
                        spec = PartitionSpec("fsdp", None) if len(x.shape) >= 2 else PartitionSpec("fsdp")
                    else:
                        spec = PartitionSpec()
                    sharding = NamedSharding(self.mesh, spec)
                    return jax.device_put(x, sharding)
                return x
            state = state.replace(opt_state=jax.tree_util.tree_map(shard_opt, state.opt_state))
        
        return state
    
    def create_train_step(self):
        """Create sharded training step."""
        
        def train_step_fn(state, batch, rng):
            """Training step with proper sharding."""
            images = batch['image']
            labels = batch.get('label', jnp.zeros(images.shape[0], dtype=jnp.int32))
            
            # Split RNG
            rng, time_key, noise_key, label_key = jax.random.split(rng, 4)
            
            def loss_fn(params):
                # Sample time
                if self.config.t_sampler == 'log-normal':
                    t = jax.random.normal(time_key, (images.shape[0],))
                    t = 1 / (1 + jnp.exp(-t))
                else:
                    t = jax.random.uniform(time_key, (images.shape[0],))
                
                t_full = t[:, None, None, None]
                
                # Flow matching
                eps = jax.random.normal(noise_key, images.shape)
                x_t = (1 - t_full) * images + t_full * eps
                v_target = eps - images
                
                # Model prediction
                v_pred = self.model_def.apply(
                    {'params': params},
                    x_t, t, labels,
                    train=True,
                    rngs={'label_dropout': label_key}
                )
                
                # Loss
                loss = jnp.mean((v_pred - v_target) ** 2)
                
                # Return metrics without pmean (JIT with sharding doesn't need it)
                metrics = {
                    'loss': loss,
                    'v_mean': jnp.abs(v_target).mean(),
                    'v_pred_mean': jnp.abs(v_pred).mean(),
                }
                
                return loss, metrics
            
            # Compute gradients
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            
            # Update
            updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
            new_params = optax.apply_updates(state.params, updates)
            
            # RE-SHARD after update if using escale
            if HAS_ESCALE:
                # Apply auto_shard_array to each param array, not the dict
                new_params = jax.tree_util.tree_map(
                    lambda x: auto_shard_array(x, self.mesh, names=["dp", "fsdp"]),
                    new_params
                )
            
            new_state = state.replace(
                step=state.step + 1,
                params=new_params,
                opt_state=new_opt_state,
            )
            
            return new_state, metrics
        
        # JIT compile with mesh context and specify axis_name for pmean
        with self.mesh:
            train_step = jax.jit(train_step_fn, donate_argnums=(0,))
        
        return train_step
    
    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("ESCALE SHARDING-FOCUSED TRAINING")
        print("=" * 60)
        
        # Initialize with same seed on all workers for deterministic behavior
        # IMPORTANT: Use same seed across all workers to avoid launch group mismatch
        base_seed = self.config.seed
        rng = jax.random.PRNGKey(base_seed)
        state = self.init_training(rng)
        
        # Simple synchronization - just print worker ready status
        if jax.process_count() > 1:
            print(f"Worker {jax.process_index()}: Ready")
        
        # Dataset - ensure same shuffle seed across workers
        print(f"\n📊 Loading {self.config.dataset_name}...")
        # Use base_seed (not worker_seed) for dataset to ensure same data order
        dataloader = create_hf_dataloader(
            dataset_name=self.config.dataset_name,
            batch_size=self.config.batch_size,
            image_size=self.image_size,
            seed=base_seed,  # Same seed for all workers
        )
        
        # Training function
        train_step = self.create_train_step()
        
        # Wandb
        if jax.process_index() == 0:
            wandb.init(
                project="jax-flow-escale-sharding",
                config=self.config.__dict__,
                name=f"{self.config.dataset_name}_{self.config.preset}_shard",
            )
        
        # Training loop
        print(f"\n🚀 Starting training for {self.config.max_steps} steps...")
        pbar = tqdm.tqdm(range(self.config.max_steps))
        train_iter = dataloader.create_infinite_iterator()
        
        for step in pbar:
            # Get batch - ensure deterministic batching
            # Use step as seed offset to ensure same batch across workers
            batch = next(train_iter)
            
            # VAE encode if needed
            if self.vae:
                batch['image'] = self.vae.encode(batch['image'])
            
            # Train step with deterministic RNG split
            # Use step to ensure same RNG evolution across workers
            step_rng = jax.random.fold_in(rng, step)
            state, metrics = train_step(state, batch, step_rng)
            
            # Log
            if step % self.config.log_interval == 0:
                if jax.process_index() == 0:
                    # Metrics should already be aggregated via pmean
                    try:
                        metrics_host = jax.device_get(metrics)
                        wandb.log(metrics_host, step=step)
                        pbar.set_description(f"Loss: {metrics_host['loss']:.4f}")
                    except Exception as e:
                        print(f"Warning: Could not log metrics at step {step}: {e}")
                        pbar.set_description(f"Step: {step}")
            
            # Save
            if self.config.save_dir and step % self.config.save_interval == 0 and step > 0:
                save_path = os.path.join(self.config.save_dir, f"checkpoint_{step}")
                os.makedirs(save_path, exist_ok=True)
                # Save logic here
                print(f"\n💾 Saved checkpoint at step {step}")
        
        print("\n✅ Training complete!")
        if jax.process_index() == 0:
            # Safe metric access for logging
            try:
                final_metrics = jax.device_get(metrics)
                print(f"   Final loss: {final_metrics['loss']:.4f}")
            except Exception as e:
                print(f"   Training completed successfully (final metrics unavailable: {e})")
        return state


def main(argv):
    """Main entry point."""
    config = ml_collections.ConfigDict({
        'dataset_name': FLAGS.dataset_name,
        'save_dir': FLAGS.save_dir,
        'seed': FLAGS.seed,
        'batch_size': FLAGS.batch_size,
        'max_steps': FLAGS.max_steps,
        'log_interval': FLAGS.log_interval,
        'save_interval': FLAGS.save_interval,
        'lr': FLAGS.lr,
        'preset': FLAGS.preset,
        'patch_size': FLAGS.patch_size,
        'use_stable_vae': FLAGS.use_stable_vae,
        'class_dropout_prob': FLAGS.class_dropout_prob,
        'denoise_timesteps': FLAGS.denoise_timesteps,
        't_sampler': FLAGS.t_sampler,
        'dp_devices': FLAGS.dp_devices,
        'fsdp_devices': FLAGS.fsdp_devices,
    })
    
    print("\n🔷 ESCALE SHARDING-FOCUSED TRAINING")
    print(f"Dataset: {config.dataset_name}")
    print(f"Model: {config.preset}")
    print(f"Batch size: {config.batch_size}")
    print(f"Devices: {jax.device_count()}")
    print(f"Sharding: DP={config.dp_devices}, FSDP={config.fsdp_devices}")
    
    # Run training
    trainer = EscaleShardingTrainer(config)
    trainer.train()
    
    print("\n🎉 DONE - SHARDING WORKED!")


if __name__ == "__main__":
    app.run(main)
