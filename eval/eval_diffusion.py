import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from tqdm import tqdm
from models.diffusion_lowdim import DiffusionMLP
from datasets.dataset import SensorDataset
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from utils.constants import FORCE_IDXS, VELOCITY_IDXS, DEFAULT_DATA_PATH
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def sample_DDPM(model, noise_scheduler, cond_inputs, gen_dims, gen_modalities, target_shape, device):
    """
    Sample from the diffusion model
    
    Args:
        model: trained diffusion model
        noise_scheduler: noise scheduler for denoising
        cond_inputs: dictionary of conditioning inputs
        gen_dims: dictionary of generation dimensions for each modality
        gen_modalities: list of modalities to generate
        target_shape: shape of the target to generate
        device: device to run on
    
    Returns:
        Generated sample
    """
    batch_size = target_shape[0]
    
    sample = torch.randn(target_shape, device=device)
    
    for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
        timesteps_float = timesteps.float().unsqueeze(-1)
        
        gen_inputs = {}
        start_idx = 0
        for modality in gen_modalities:
            modality_dim = gen_dims[modality]
            gen_inputs[modality] = sample[:, start_idx:start_idx + modality_dim]
            start_idx += modality_dim
        
        model_outputs = model(gen_inputs, cond_inputs, timesteps_float)
        noise_pred = torch.cat([pred for pred in model_outputs.values()], dim=-1)
        sample = noise_scheduler.step(noise_pred, t, sample).prev_sample
    
    return sample


def split_sample_by_modalities(sample, gen_dims, gen_modalities):
    """
    Split a concatenated sample back into individual modality components.
    
    Args:
        sample: concatenated sample tensor
        gen_dims: dictionary of generation dimensions for each modality
        gen_modalities: list of modalities to generate
    
    Returns:
        dictionary of modality-specific samples
    """
    gen_samples = {}
    start_idx = 0
    for modality in gen_modalities:
        modality_dim = gen_dims[modality]
        gen_samples[modality] = sample[:, start_idx:start_idx + modality_dim]
        start_idx += modality_dim
    return gen_samples


def compute_modality_losses(generated_sample, target_sample, gen_dims, gen_modalities):
    """
    Compute MSE loss for each modality separately.
    
    Args:
        generated_sample: generated sample tensor
        target_sample: target sample tensor
        gen_dims: dictionary of generation dimensions for each modality
        gen_modalities: list of modalities to generate
    
    Returns:
        dictionary of modality-specific losses
    """
    gen_split = split_sample_by_modalities(generated_sample, gen_dims, gen_modalities)
    target_split = split_sample_by_modalities(target_sample, gen_dims, gen_modalities)
    
    modality_losses = {}
    for modality in gen_modalities:
        modality_losses[modality] = F.mse_loss(gen_split[modality], target_split[modality]).item()
    
    return modality_losses

@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg: DictConfig):
    checkpoint_path = cfg.checkpoint_path
    checkpoint_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    hydra_config_path = os.path.join(checkpoint_dir, ".hydra", "config.yaml")
    if os.path.exists(hydra_config_path):
        print(f"Loading config from checkpoint directory: {hydra_config_path}")
        loaded_cfg = OmegaConf.load(hydra_config_path)
        with open_dict(loaded_cfg):
            loaded_cfg.checkpoint_path = checkpoint_path
        cfg = OmegaConf.merge(loaded_cfg, cfg)

    # config stuff
    batch_size = cfg.batch_size
    pred_horizon = cfg.pred_horizon
    force_dim = cfg.force_dim
    velocity_dim = cfg.velocity_dim
    action_dim = cfg.action_dim * pred_horizon
    state_dim = cfg.state_dim
    hidden_dim = cfg.hidden_dim
    num_layers = cfg.num_layers
    t_embed_dim = cfg.t_embed_dim
    dropout = cfg.dropout
    fusion_method = cfg.get('fusion_method', 'concat') 
    device = cfg.device if torch.cuda.is_available() else 'cpu'

    cond_modalities = cfg.get('cond_modalities', ['force', 'velocity', 'action'])
    gen_modalities = cfg.get('gen_modalities', ['force', 'velocity', 'state'])
    
    future_cond_modalities = cfg.get('future_cond_modalities', [])

    print(OmegaConf.to_yaml(cfg))

    all_modalities = list(set(cond_modalities + gen_modalities))
    
    dataset = SensorDataset(
        data_path=cfg.get('data_path', DEFAULT_DATA_PATH), 
        pred_horizon=pred_horizon, 
        eval_mode=True,
        modalities=all_modalities
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    gen_dims = {}
    cond_dims = {}
    
    modality_dims = {
        'force': force_dim,
        'velocity': velocity_dim,
        'state': state_dim,
        'action': action_dim,
        'future_force': force_dim,  
        'future_velocity': velocity_dim
    }
    
    for modality in gen_modalities:
        if modality in modality_dims:
            gen_dims[modality] = modality_dims[modality]
    
    for modality in cond_modalities:
        if modality in modality_dims:
            cond_dims[modality] = modality_dims[modality]
    
    for modality in future_cond_modalities:
        if modality in modality_dims:
            cond_dims[modality] = modality_dims[modality]

    model = DiffusionMLP(
        gen_dims=gen_dims,
        cond_dims=cond_dims,
        hidden_dim=hidden_dim,
        num_blocks=num_layers,
        t_embed_dim=t_embed_dim,
        dropout=dropout,
        return_dict=True,
        fusion_method=fusion_method
    ).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.6f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict (no metadata available)")
    
    model.eval()

    wandb.init(project="sensorfusion-diffusion-eval",
               name=f"eval-{cfg.wandb.name}",
               config=OmegaConf.to_container(cfg, resolve=True))

    total_loss = 0.0
    num_batches = 0
    modality_losses_history = []
    modality_losses_sum = defaultdict(float)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            cond_inputs = {}
            for modality in cond_modalities:
                if modality == 'action':
                    # actions need to be flattened
                    cond_inputs[modality] = batch['action_sequence'].to(device).reshape(batch_size, -1)
                else:
                    cond_inputs[modality] = batch[f'{modality}'].to(device)
            
            for modality in future_cond_modalities:
                if modality == 'future_force':
                    cond_inputs[modality] = batch['target_force'][:, -1, :].to(device)
                elif modality == 'future_velocity':
                    cond_inputs[modality] = batch['target_velocity'][:, -1, :].to(device)
                else:
                    cond_inputs[modality] = batch[f'target_{modality.replace("future_", "")}'][:, -1, :].to(device)
            
            gen_targets = {}
            for modality in gen_modalities:
                if modality == 'action':
                    gen_targets[modality] = batch[f'{modality}_sequence'].to(device).reshape(batch_size, -1)
                else:
                    gen_targets[modality] = batch[f'target_{modality}'][:, -1, :].to(device).reshape(batch_size, -1)
            
            x_target = torch.cat([gen_targets[mod] for mod in gen_modalities], dim=-1)
            
            sample = sample_DDPM(model, noise_scheduler, cond_inputs, gen_dims, gen_modalities, x_target.shape, device)

            loss = F.mse_loss(sample, x_target)
            total_loss += loss.item()
            
            batch_modality_losses = compute_modality_losses(sample, x_target, gen_dims, gen_modalities)
            modality_losses_history.append(batch_modality_losses)
            
            for modality, mod_loss in batch_modality_losses.items():
                modality_losses_sum[modality] += mod_loss
            
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_modality_losses = {mod: modality_losses_sum[mod] / num_batches for mod in gen_modalities}
    
    for modality in gen_modalities:
        print(f"  {modality} (dim: {gen_dims[modality]}): {avg_modality_losses[modality]:.6f}")
    
    wandb.log({"eval_loss": avg_loss})
    for modality, mod_loss in avg_modality_losses.items():
        wandb.log({f"eval_loss_{modality}": mod_loss})

if __name__ == "__main__":
    main()
