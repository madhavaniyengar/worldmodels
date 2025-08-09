import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from diffusers import DDPMScheduler
from tqdm import tqdm
from models.diffusion_lowdim import DiffusionMLP
from datasets.dataset import SensorDataset
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from utils.constants import FORCE_IDXS, VELOCITY_IDXS
from eval.eval_diffusion import sample_DDPM
from utils.constants import DEFAULT_DATA_PATH

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, filename=None):
    """
    Save a checkpoint containing model state, optimizer state, and training metadata.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoints
        filename: Optional custom filename, if None uses epoch number
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pt"
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Load a checkpoint and restore model and optimizer states.
    
    Args:
        model: The model to load state into
        optimizer: The optimizer to load state into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
    
    Returns:
        tuple: (epoch, loss) from the checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.6f}")
    return epoch, loss


@hydra.main(version_base=None, config_path="../conf", config_name="diffusion_lowdim")
def main(cfg):
    # config stuffs
    epochs = cfg.epochs
    pred_horizon = cfg.pred_horizon
    batch_size = cfg.batch_size
    learning_rate = cfg.learning_rate
    force_dim = cfg.force_dim
    velocity_dim = cfg.velocity_dim
    action_dim = cfg.action_dim * pred_horizon
    state_dim = cfg.state_dim
    hidden_dim = cfg.hidden_dim
    num_layers = cfg.num_layers
    t_embed_dim = cfg.t_embed_dim
    dropout = cfg.dropout
    fusion_method = cfg.fusion_method
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    
    # modalities configuration
    cond_modalities = cfg.get('cond_modalities', None)
    gen_modalities = cfg.get('gen_modalities', None)
    
    future_cond_modalities = cfg.get('future_cond_modalities', [])
    
    # checkpointing 
    save_checkpoint_every = cfg.get('save_checkpoint_every', 10)  
    checkpoint_dir = os.path.join(HydraConfig.get().run.dir, cfg.get('checkpoint_dir', 'checkpoints'))
    resume_from_checkpoint = cfg.get('resume_from_checkpoint', None)
    
    debug = cfg.get('debug', False)

    print(OmegaConf.to_yaml(cfg))
    

    all_modalities = list(set(cond_modalities + gen_modalities))
    
    dataset = SensorDataset(
        data_path=cfg.get('data_path', DEFAULT_DATA_PATH), 
        pred_horizon=pred_horizon, 
        eval_mode=False,
        modalities=all_modalities
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    start_epoch = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        start_epoch, _ = load_checkpoint(model, optimizer, resume_from_checkpoint, device)
        start_epoch += 1
        print(f"Resuming training from epoch {start_epoch}")
        
    print(f"\n=== Modality Configuration ===")
    print(f"Conditioning modalities: {cond_modalities}")
    print(f"Future conditioning modalities: {future_cond_modalities}")
    print(f"Generation modalities: {gen_modalities}")
    print(f"All modalities needed: {all_modalities}")
    print(f"Generation dimensions: {gen_dims}")
    print(f"Conditioning dimensions: {cond_dims}")
    print("=" * 30 + "\n")

    # wandb
    if not debug:
        wandb.init(project="sensorfusion-diffusion",
                name=f"{cfg.wandb.name}",
                config=OmegaConf.to_container(cfg, resolve=True)
        )
    else:
        print("Debug mode enabled - wandb logging disabled")
    

    validation_freq = cfg.get('validation_freq', 10)
    best_val_loss = float('inf')
    patience = cfg.get('patience', 50)
    patience_counter = 0

    # training
    for epoch in tqdm(range(start_epoch, epochs), desc="Training"):
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            cond_inputs = {}
            for modality in cond_modalities:
                if modality == 'action':
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
            
            noise = torch.randn_like(x_target)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (x_target.size(0),), device=device).long()
            noisy_input = noise_scheduler.add_noise(x_target, noise, timesteps)

            timesteps_float = timesteps.float().unsqueeze(-1)

            gen_inputs = {}
            start_idx = 0
            for modality in gen_modalities:
                modality_dim = gen_dims[modality]
                gen_inputs[modality] = noisy_input[:, start_idx:start_idx + modality_dim]
                start_idx += modality_dim

            model_outputs = model(gen_inputs, cond_inputs, timesteps_float)
            noise_pred = torch.cat([pred for pred in model_outputs.values()], dim=-1)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        if not debug:
            wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})
        else:
            print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.6f}")
        
        # VAL
        if (epoch + 1) % validation_freq == 0:
            model.eval()
            val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for val_batch in tqdm(val_dataloader, desc=f"Validating epoch {epoch+1}", leave=False):
                    cond_inputs = {}
                    for modality in cond_modalities:
                        if modality == 'action':
                            cond_inputs[modality] = val_batch['action_sequence'].to(device).reshape(batch_size, -1)
                        else:
                            cond_inputs[modality] = val_batch[f'{modality}'].to(device)
                    
                    for modality in future_cond_modalities:
                        if modality == 'future_force':
                            cond_inputs[modality] = val_batch['target_force'][:, -1, :].to(device)
                        elif modality == 'future_velocity':
                            cond_inputs[modality] = val_batch['target_velocity'][:, -1, :].to(device)
                        else:
                            cond_inputs[modality] = val_batch[f'target_{modality.replace("future_", "")}'][:, -1, :].to(device)
                    
                    gen_targets = {}
                    for modality in gen_modalities:
                        if modality == 'action':
                            gen_targets[modality] = val_batch[f'{modality}_sequence'].to(device).reshape(batch_size, -1)
                        else:
                            gen_targets[modality] = val_batch[f'target_{modality}'][:, -1, :].to(device).reshape(batch_size, -1)
                    
                    x_target = torch.cat([gen_targets[mod] for mod in gen_modalities], dim=-1)
                    
                    sample = sample_DDPM(model, noise_scheduler, cond_inputs, gen_dims, gen_modalities, x_target.shape, device)
                    
                    loss = F.mse_loss(sample, x_target)
                    val_loss += loss.item()
                    num_val_batches += 1
            
            avg_val_loss = val_loss / num_val_batches
            if not debug:
                wandb.log({"val_loss": avg_val_loss, "epoch": epoch + 1})
            else:
                print(f"Epoch {epoch + 1}, Val Loss: {avg_val_loss:.6f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch + 1, avg_loss, checkpoint_dir, "best_model.pt")
            else:
                patience_counter += validation_freq
            
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            model.train()
        
        if (epoch + 1) % save_checkpoint_every == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, checkpoint_dir)

if __name__ == "__main__":
    main()
