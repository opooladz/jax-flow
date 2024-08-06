try:
    from localutils.debugger import enable_debug
    enable_debug()
except:
    pass

from typing import Any
import jax.numpy as jnp
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import flax
import optax
import wandb
from ml_collections import config_flags
import ml_collections
import matplotlib.pyplot as plt

from utils.wandb import setup_wandb, default_wandb_config
from utils.train_state import TrainStateEps
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from utils.sharding import create_sharding
from utils.datasets import get_dataset
from utils.spectral_optimizer import spectral_init, scale_spectral_norm
from model import DiT

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'imagenet256', 'Environment name.')
flags.DEFINE_string('load_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string('save_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string('fid_stats', None, 'FID stats file.')
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 200000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1_000_000), 'Number of training steps.')
flags.DEFINE_integer('debug_overfit', 0, 'Debug overfitting.')

model_config = ml_collections.ConfigDict({
    # Make sure to run with Large configs when we actually want to run!
    'lr': 0.05,
    'lr_scale_patch': 1.0,
    'lr_scale_embed': 1.0,
    'lr_scale_final': 1.0,
    'lr_scale_time': 1.0,
    'beta1': 0.9,
    'beta2': 0.999,
    'weight_decay': 0.0,
    'warmup': 0,
    'use_spectral_norm': 1,
    'hidden_size': 64, 
    'patch_size': 2, 
    'depth': 4,
    'num_heads': 4,
    'mlp_ratio': 4,
    'dropout': 0.0,
    'class_dropout_prob': 0.1,
    'num_classes': 1000,
    'denoise_timesteps': 32,
    'cfg_scale': 4.0,
    'target_update_rate': 0.9999,
    't_sampler': 'uniform',
    't_conditioning': 1,
    'preset': 'none',
    'use_stable_vae': 1,
    'sharding': 'dp', # or 'fsdp'.
    'vae_type': 'sd', # or 'sdxl'.
    'random_fid_labels': 0,
    'augmentation': 0,
})

wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'flow',
    'name': 'flow_{dataset_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('model', model_config, lock_config=False)
    
##############################################
## Training Code.
##############################################
def main(_):

    np.random.seed(FLAGS.seed)
    print("Using devices", jax.local_devices())
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    num_hosts = global_device_count // device_count
    print("Device count", device_count)
    print("Global device count", global_device_count)
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)
    print("Global Batch: ", FLAGS.batch_size)
    print("Node Batch: ", local_batch_size)
    print("Device Batch:", local_batch_size // device_count)

    # Create wandb logger
    if jax.process_index() == 0:
        setup_wandb(FLAGS.model.to_dict(), **FLAGS.wandb)

    dataset = get_dataset(FLAGS.dataset_name, local_batch_size, True, FLAGS.debug_overfit)
    dataset_valid = get_dataset(FLAGS.dataset_name, local_batch_size, False, FLAGS.debug_overfit)
    example_obs, example_labels = next(dataset)
    example_obs = example_obs[:1]
    example_obs_shape = example_obs.shape

    if FLAGS.model.use_stable_vae:
        vae = StableVAE.create(vae_type=FLAGS.model.vae_type)
        example_obs = vae.encode(jax.random.PRNGKey(0), example_obs)
        example_obs_shape = example_obs.shape
        vae_rng = jax.random.PRNGKey(42)
        vae_encode = jax.jit(vae.encode)
        vae_decode = jax.jit(vae.decode)
        
    if FLAGS.fid_stats is not None:
        from utils.fid import get_fid_network, fid_from_stats
        get_fid_activations = get_fid_network() 
        truth_fid_stats = np.load(FLAGS.fid_stats)

    ###################################
    # Creating Model and put on devices.
    ###################################
    FLAGS.model.image_channels = example_obs_shape[-1]
    FLAGS.model.image_size = example_obs_shape[1]
    dit_args = {
        'patch_size': FLAGS.model['patch_size'],
        'hidden_size': FLAGS.model['hidden_size'],
        'depth': FLAGS.model['depth'],
        'num_heads': FLAGS.model['num_heads'],
        'mlp_ratio': FLAGS.model['mlp_ratio'],
        'class_dropout_prob': FLAGS.model['class_dropout_prob'],
        'num_classes': FLAGS.model['num_classes'],
        'dropout': FLAGS.model['dropout'],
        'use_spectral_norm': FLAGS.model['use_spectral_norm'],
        'lr_scale_patch': FLAGS.model['lr_scale_patch'],
        'lr_scale_embed': FLAGS.model['lr_scale_embed'],
        'lr_scale_final': FLAGS.model['lr_scale_final'],
        'lr_scale_time': FLAGS.model['lr_scale_time'],
    }
    model_def = DiT(**dit_args)
    tabulate_fn = flax.linen.tabulate(model_def, jax.random.PRNGKey(0))
    print(tabulate_fn(example_obs, jnp.zeros((1,)), jnp.zeros((1,), dtype=jnp.int32)))

    lr = FLAGS.model['lr'] if FLAGS.model.warmup == 0 else optax.linear_schedule(0.0, FLAGS.model['lr'], FLAGS.model['warmup'])
    adam = optax.adamw(learning_rate=lr, b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'], weight_decay=FLAGS.model['weight_decay'])
    # if FLAGS.model.use_spectral_norm:
    #     tx = optax.chain(adam, scale_spectral_norm())
    # else:
    tx = optax.chain(adam)
    
    def init(rng):
        param_key, dropout_key, dropout2_key = jax.random.split(rng, 3)
        example_t = jnp.zeros((1,))
        example_label = jnp.zeros((1,), dtype=jnp.int32)
        example_obs = jnp.zeros(example_obs_shape)
        model_rngs = {'params': param_key, 'label_dropout': dropout_key, 'dropout': dropout2_key}
        params = model_def.init(model_rngs, example_obs, example_t, example_label)['params']
        opt_state = tx.init(params)
        return TrainStateEps.create(model_def, params, rng=rng, tx=tx, opt_state=opt_state)
    
    # Sharded Parameters (either replicated, or fsdp).
    rng = jax.random.PRNGKey(FLAGS.seed)
    train_state_shape = jax.eval_shape(init, rng)
    data_sharding, train_state_sharding, no_shard, shard_data, global_to_local = create_sharding(FLAGS.model.sharding, train_state_shape)
    train_state = jax.jit(init, out_shardings=train_state_sharding)(rng)
    # jax.debug.visualize_array_sharding(train_state.params['FinalLayer_0']['Dense_0']['kernel'].value)

    if FLAGS.load_dir is not None:
        cp = Checkpoint(FLAGS.load_dir)
        replace_dict = cp.load_as_dict()['train_state']
        train_state = train_state.replace(**replace_dict)
        train_state = jax.jit(lambda x : x, out_shardings=train_state_sharding)(train_state)
        print("Loaded model with step", train_state.step)
        # jax.debug.visualize_array_sharding(train_state.params['FinalLayer_0']['Dense_0']['kernel'].value)
        del cp

    visualize_labels = example_labels[:global_device_count]
    visualize_labels = shard_data(visualize_labels)
    visualize_labels = jax.experimental.multihost_utils.process_allgather(visualize_labels)
    imagenet_labels = open('data/imagenet_labels.txt').read().splitlines()

    ###################################
    # Helpers
    ###################################

    def get_x_t(images, eps, t):
        x_0 = eps # Noise
        x_1 = images # Data
        t = jnp.clip(t, 0, 1-0.01) # Always include a little bit of noise.
        return (1-t) * x_0 + t * x_1

    def get_v(images, eps):
        x_0 = eps
        x_1 = images
        return x_1 - x_0

    ###################################
    # Update Function
    ###################################
    @partial(jax.jit, out_shardings=(train_state_sharding, no_shard))
    def update(train_state, images, labels):
        new_rng, label_key, dropout_key, time_key, noise_key = jax.random.split(train_state.rng, 5)

        def loss_fn(grad_params):
            # Sample a t for training.
            if FLAGS.model['t_sampler'] == 'log-normal':
                t = jax.random.normal(time_key, (images.shape[0],))
                t = ((1 / (1 + jnp.exp(-t))))
            elif FLAGS.model['t_sampler'] == 'uniform':
                t = jax.random.uniform(time_key, (images.shape[0],), minval=0, maxval=1)
            elif FLAGS.model['t_sampler'] == 'permutation':
                t = jnp.arange(images.shape[0]) / images.shape[0]
                t = jax.random.permutation(time_key, t)
            elif FLAGS.model['t_sampler'] == 'discrete':
                t = jax.random.randint(time_key, (images.shape[0],), 0, FLAGS.model['denoise_timesteps']) / FLAGS.model['denoise_timesteps']
            elif FLAGS.model['t_sampler'] == 'debug-constant':
                t = jnp.ones((images.shape[0],)) * 0.5
            t_full = t[:, None, None, None] # [batch, 1, 1, 1]
            eps = jax.random.normal(noise_key, images.shape)
            x_t = get_x_t(images, eps, t_full)
            v_t = get_v(images, eps)

            if FLAGS.model['t_conditioning'] == 0:
                t = jnp.zeros_like(t)
            
            rngs = {'label_dropout': label_key, 'dropout': dropout_key}
            v_prime, activations = train_state.call_model(x_t, t, labels, train=True, rngs=rngs, 
                                                          params=grad_params, return_activations=True)
            loss = jnp.mean((v_prime - v_t) ** 2)
            
            return loss, {
                'l2_loss': loss,
                'v_abs_mean': jnp.abs(v_t).mean(),
                'v_pred_abs_mean': jnp.abs(v_prime).mean(),
                **{'activations/' + k : jnp.mean(jnp.abs(v)) for k, v in activations.items()}
            }
        
        grads, info = jax.grad(loss_fn, has_aux=True)(train_state.params)
        updates, new_opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
        new_params = optax.apply_updates(train_state.params, updates)

        info['grad_norm'] = optax.global_norm(grads)
        info['update_norm'] = optax.global_norm(updates)
        info['param_norm'] = optax.global_norm(new_params)
        info['norms/final_layer'] = optax.global_norm(new_params['FinalLayer_0']['Dense_0']['kernel'])
        info['norms/patch'] = optax.global_norm(new_params['PatchEmbed_0']['Conv_0']['kernel'])
        info['norms/embed'] = optax.global_norm(new_params['LabelEmbedder_0']['Embed_0']['embedding'])

        train_state = train_state.replace(rng=new_rng, step=train_state.step + 1, params=new_params, opt_state=new_opt_state)
        train_state = train_state.update_eps(FLAGS.model['target_update_rate'])
        return train_state, info

    ###################################
    # Train Loop
    ###################################

    def eval_model(train_state):
        with jax.spmd_mode('allow_all'):
            # Needs to be in a separate function so garbage collection works correctly.
            def process_img(img):
                if FLAGS.model.use_stable_vae:
                    img = vae_decode(img[None])[0]
                img = img * 0.5 + 0.5
                img = jnp.clip(img, 0, 1)
                img = np.array(img)
                return img

            print("Valid loss")
            # Validation Losses
            valid_images, valid_labels = next(dataset_valid)
            valid_images_sharded, valid_labels_sharded = shard_data(valid_images, valid_labels)
            if FLAGS.model.use_stable_vae:
                valid_images_sharded = vae_encode(vae_rng, valid_images_sharded)
                valid_images = vae_encode(vae_rng, valid_images)
            _, valid_update_info = update(train_state, valid_images_sharded, valid_labels_sharded)
            valid_update_info = jax.device_get(valid_update_info)
            valid_update_info = jax.tree_map(lambda x: x.mean(), valid_update_info)
            valid_metrics = {f'validation/{k}': v for k, v in valid_update_info.items()}
            visualize_images_shape = valid_images[:global_device_count].shape
            if jax.process_index() == 0:
                wandb.log(valid_metrics, step=i)

            @partial(jax.jit, static_argnums=(4, 5))
            def call_model(train_state, images, t, labels, use_cfg, cfg_scale):
                print("Call_model with shapes", images.shape, t.shape, labels.shape)
                if not use_cfg:
                    return train_state.call_model_eps(images, t, labels, train=False, force_drop_ids=False)
                else:
                    labels_uncond = jnp.ones(labels.shape, dtype=jnp.int32) * FLAGS.model['num_classes'] # Null token
                    v_pred_uncond = train_state.call_model_eps(images, t, labels_uncond, train=False, force_drop_ids=False)
                    v_pred_label = train_state.call_model_eps(images, t, labels, train=False, force_drop_ids=False)
                    v = v_pred_uncond + cfg_scale * (v_pred_label - v_pred_uncond)
                    return v

            print("Training Loss")
            # Training loss on various t.
            if FLAGS.model.use_stable_vae:
                batch_images_enc = vae_encode(vae_rng, batch_images)
            else:
                batch_images_enc = batch_images
            mse_total = []
            for t in np.arange(0, 11):
                key = jax.random.PRNGKey(42)
                t = t / 10
                t_full = jnp.full((batch_images_enc.shape), t)
                t_vector = jnp.full((batch_images_enc.shape[0],), t)
                eps = jax.random.normal(key, batch_images_enc.shape)
                x_t = get_x_t(batch_images_enc, eps, t_full)
                v = get_v(batch_images_enc, eps)
                x_t, t_vector = shard_data(x_t, t_vector)
                pred_v = call_model(train_state, x_t, t_vector, batch_labels_sharded, False, 0.0)
                pred_v = global_to_local(pred_v)
                assert pred_v.shape == v.shape
                mse_loss = jnp.mean((v - pred_v) ** 2)
                mse_total.append(mse_loss)
                if jax.process_index() == 0:
                    wandb.log({f'training_loss_t/{t}': mse_loss}, step=i)
            mse_total = jnp.array(mse_total[1:-1])
            if jax.process_index() == 0:
                wandb.log({'training_loss_t/mean': mse_total.mean()}, step=i)

            print("Validation Loss")
            # Validation loss on various t.
            mse_total = []
            fig, axs = plt.subplots(3, 10, figsize=(30, 20))
            for t in np.arange(0, 11):
                key = jax.random.PRNGKey(42)
                t = t / 10
                t_full = jnp.full((valid_images.shape), t)
                t_vector = jnp.full((valid_images.shape[0],), t)
                eps = jax.random.normal(key, valid_images.shape)
                x_t = get_x_t(valid_images, eps, t_full)
                v = get_v(valid_images, eps)
                x_t, t_vector = shard_data(x_t, t_vector)
                pred_v = call_model(train_state, x_t, t_vector, valid_labels_sharded, False, 0.0)
                pred_v = global_to_local(pred_v)
                assert pred_v.shape == v.shape
                mse_loss = jnp.mean((v - pred_v) ** 2)
                mse_total.append(mse_loss)
                if jax.process_index() == 0:
                    wandb.log({f'validation_loss_t/{t}': mse_loss}, step=i)
            mse_total = jnp.array(mse_total[1:-1])
            if jax.process_index() == 0:
                wandb.log({'validation_loss_t/mean': mse_total.mean()}, step=i)
                plt.close(fig)

            print("One-step Denoising at various t.")
            if len(jax.local_devices()) == 8:
                t = jnp.arange(8) / 8 # between 0 and 0.875
                t = jnp.repeat(t, valid_images.shape[0] // 8, axis=0) # [8, batch//devices, etc..] DEVICES=8
                key = jax.random.PRNGKey(42)
                eps = jax.random.normal(key, valid_images.shape)
                x_t = get_x_t(valid_images, eps, t[..., None, None, None])
                x_t, t = shard_data(x_t, t)
                v_pred = call_model(train_state, x_t, t, valid_labels_sharded, False, 0.0)
                x_1_pred = x_t + v_pred * (1-t[..., None, None, None])
                x_t = jax.experimental.multihost_utils.process_allgather(x_t)
                x_1_pred = jax.experimental.multihost_utils.process_allgather(x_1_pred)
                valid_images_sharded_gather = jax.experimental.multihost_utils.process_allgather(valid_images_sharded)
                if jax.process_index() == 0:
                    # plot comparison witah matplotlib. put each reconstruction side by side.
                    fig, axs = plt.subplots(8, 8*3, figsize=(90, 30))
                    for j in range(min(8, valid_images_sharded_gather.shape[0] // 8)):
                        for k in range(8):
                            axs[j,3*k].imshow(process_img(valid_images_sharded_gather[j*8 + k]), vmin=0, vmax=1)
                            axs[j,3*k+1].imshow(process_img(x_t[j*8 + k]), vmin=0, vmax=1)
                            axs[j,3*k+2].imshow(process_img(x_1_pred[j*8 + k]), vmin=0, vmax=1)
                    wandb.log({f'reconstruction_n': wandb.Image(fig)}, step=i)
                    plt.close(fig)

            print("Full Denoising with different CFG")
            # Full Denoising with different CFG;
            key = jax.random.PRNGKey(42 + jax.process_index() + i)
            eps = jax.random.normal(key, visualize_images_shape) # [devices, batch//devices, etc..]
            delta_t = 1.0 / FLAGS.model.denoise_timesteps
            for cfg_scale in [None, 0, 1.5, 4]:
                x = eps
                x = shard_data(x)
                all_x = []
                for ti in range(FLAGS.model.denoise_timesteps):
                    print(ti)
                    t = ti / FLAGS.model.denoise_timesteps # From x_0 (noise) to x_1 (data)
                    t_vector = jnp.full((visualize_images_shape[0],), t)
                    t_vector = shard_data(t_vector)
                    if cfg_scale is None:
                        v = call_model(train_state, x, t_vector, visualize_labels, False, 0.0)
                    else:
                        v = call_model(train_state, x, t_vector, visualize_labels, True, cfg_scale)
                    x = x + v * delta_t
                    if ti % (FLAGS.model.denoise_timesteps // 8) == 0 or ti == FLAGS.model.denoise_timesteps-1:
                        np_x = jax.experimental.multihost_utils.process_allgather(x)
                        all_x.append(np.array(np_x))
                all_x = np.stack(all_x, axis=1) # [devices, batch//devices, timesteps, etc..]
                all_x = all_x[:, -8:]

                if jax.process_index() == 0:
                    # plot comparison witah matplotlib. put each reconstruction side by side.
                    fig, axs = plt.subplots(8, 8, figsize=(30, 30))
                    for j in range(8):
                        for t in range(8):
                            axs[t, j].imshow(process_img(all_x[j, t]), vmin=0, vmax=1)
                        axs[0, j].set_title(f"{imagenet_labels[visualize_labels[j]]}")
                    wandb.log({f'sample_cfg_{cfg_scale}': wandb.Image(fig)}, step=i)
                    plt.close(fig)

            print("Denoising at N steps")
            # Denoising at different numbers of steps.
            key = jax.random.PRNGKey(42 + jax.process_index() + i)
            eps = jax.random.normal(key, visualize_images_shape) # [devices, batch//devices, etc..]
            for denoise_timesteps in [1, 4, 32]:
                delta_t = 1.0 / denoise_timesteps
                x = eps
                x = shard_data(x)
                for ti in range(denoise_timesteps):
                    t = ti / denoise_timesteps # From x_0 (noise) to x_1 (data)
                    t_vector = jnp.full((visualize_images_shape[0],), t)
                    t_vector = shard_data(t_vector)
                    v = call_model(train_state, x, t_vector, visualize_labels, True, FLAGS.model.cfg_scale)
                    x = x + v * delta_t
                x = jax.experimental.multihost_utils.process_allgather(x)
                if jax.process_index() == 0:
                    # plot comparison witah matplotlib. put each reconstruction side by side.
                    fig, axs = plt.subplots(8, 8, figsize=(30, 30))
                    for j in range(min(8, x.shape[0] // 8)):
                        for t in range(8):
                            axs[t, j].imshow(process_img(x[j*8 + t]), vmin=0, vmax=1)
                        axs[0, j].set_title(f"{imagenet_labels[visualize_labels[j]]}")
                    wandb.log({f'sample_N/{denoise_timesteps}': wandb.Image(fig)}, step=i)
                    plt.close(fig)

            print("FID calc")
            # # FID calculation.
            if FLAGS.fid_stats is not None:
                cfg_list = [0] if FLAGS.model.cfg_scale == 0 else [-1, 0, 1.5, 4]
                for cfg_scale in cfg_list:
                    activations = []
                    valid_images_shape = valid_images.shape
                    num_generations = 4096
                    for fid_it in tqdm.tqdm(range(num_generations // FLAGS.batch_size)):
                        _, valid_labels = next(dataset_valid)
                        if FLAGS.model.random_fid_labels:
                            valid_labels = jax.random.randint(jax.random.PRNGKey(42 + fid_it), (FLAGS.batch_size,), 0, FLAGS.model.num_classes)
                        valid_labels = shard_data(valid_labels)
                        key = jax.random.PRNGKey(42 + fid_it)
                        x = jax.random.normal(key, valid_images_shape)
                        x = shard_data(x)
                        delta_t = 1.0 / FLAGS.model.denoise_timesteps
                        for ti in range(FLAGS.model.denoise_timesteps):
                            t = ti / FLAGS.model.denoise_timesteps # From x_0 (noise) to x_1 (data)
                            t_vector = jnp.full((valid_images_shape[0], ), t)
                            t_vector = shard_data(t_vector)
                            if cfg_scale == -1:
                                v = call_model(train_state, x, t_vector, valid_labels, False, 0.0)
                            else:
                                v = call_model(train_state, x, t_vector, valid_labels, True, cfg_scale)
                            x = x + v * delta_t
                        if FLAGS.model.use_stable_vae:
                            x = vae_decode(x)
                        x = jax.image.resize(x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
                        x = 2 * x - 1
                        acts = get_fid_activations(x)[..., 0, 0, :] # [devices, batch//devices, 2048]
                        acts = jax.experimental.multihost_utils.process_allgather(acts)
                        acts = np.array(acts)
                        activations.append(acts)
                    if jax.process_index() == 0:
                        activations = np.concatenate(activations, axis=0)
                        activations = activations.reshape((-1, activations.shape[-1]))
                        mu1 = np.mean(activations, axis=0)
                        sigma1 = np.cov(activations, rowvar=False)
                        fid = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
                        wandb.log({f'fid_cfg{cfg_scale}': fid}, step=i)
                        print("FID cfg", cfg_scale, ":", fid)
            
        del valid_images, valid_labels
        del x, x_t, eps
        print("Finished all the eval stuff")

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):

        if not FLAGS.debug_overfit or i == 1:
            batch_images, batch_labels = next(dataset)
            batch_images_sharded, batch_labels_sharded = shard_data(batch_images, batch_labels)
            if FLAGS.model.use_stable_vae:
                batch_images_sharded = vae_encode(vae_rng, batch_images_sharded)
        
        train_state, update_info = update(train_state, batch_images_sharded, batch_labels_sharded)

        if i % FLAGS.log_interval == 0:
            update_info = jax.device_get(update_info)
            update_info = jax.tree_map(lambda x: np.array(x), update_info)
            update_info = jax.tree_map(lambda x: x.mean(), update_info)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}

            if jax.process_index() == 0:
                print(train_metrics)
                wandb.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_model(train_state)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            train_state_gather = jax.experimental.multihost_utils.process_allgather(train_state)
            if jax.process_index() == 0:
                cp = Checkpoint(FLAGS.save_dir, parallel=False)
                cp.train_state = train_state_gather
                cp.wandb_id = wandb.run.id # For resuming the run.
                cp.save()
                del cp
            del train_state_gather

if __name__ == '__main__':
    app.run(main)