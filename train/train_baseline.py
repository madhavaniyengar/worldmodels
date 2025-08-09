import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from models.baseline_mlp import BaselineMLP
from datasets.dataset import SensorDataset
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from utils.constants import DEFAULT_DATA_PATH

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, filename=None):
    """
    Save a checkpoint containing model state, optimizer state, and training metadata.
    
    Args:
        model: model to save
        optimizer: optimizer to save
        epoch: current epoch number
        loss: current loss value
        checkpoint_dir: directory to save checkpoints
        filename: optional custom filename, if None uses epoch number
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
        model: model to load state into
        optimizer: optimizer to load state into
        checkpoint_path: path to the checkpoint file
        device: device to load the model on
    
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


@hydra.main(version_base=None, config_path="../conf", config_name="baseline")
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
    dropout = cfg.dropout
    fusion_method = cfg.fusion_method
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    
    cond_modalities = cfg.get('cond_modalities', None)
    gen_modalities = cfg.get('gen_modalities', None)
    
    future_cond_modalities = cfg.get('future_cond_modalities', [])
    
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

    model = BaselineMLP(
        gen_dims=gen_dims,
        cond_dims=cond_dims,
        hidden_dim=hidden_dim,
        num_blocks=num_layers,
        dropout=dropout,
        return_dict=True,
        fusion_method=fusion_method
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # resume from checkpoint if specified
    start_epoch = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        start_epoch, _ = load_checkpoint(model, optimizer, resume_from_checkpoint, device)
        start_epoch += 1
        print(f"Resuming training from epoch {start_epoch}")
        
    if not debug:
        wandb.init(project="sensorfusion-baseline",
                name=f"{cfg.wandb.name}",
                config=OmegaConf.to_container(cfg, resolve=True)
        )
    else:
        print("DEBUIG")
    
    validation_freq = cfg.get('validation_freq', 10)
    best_val_loss = float('inf')
    patience = cfg.get('patience', 50)
    patience_counter = 0

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
            
            model_outputs = model(cond_inputs)
            
            loss = 0.0
            for modality in gen_modalities:
                loss += F.mse_loss(model_outputs[modality], gen_targets[modality])
            
            loss = loss / len(gen_modalities)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        if not debug:
            wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})
        else:
            print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.6f}")
        
        # validation
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
                    
                    model_outputs = model(cond_inputs)
                    
                    batch_loss = 0.0
                    for modality in gen_modalities:
                        batch_loss += F.mse_loss(model_outputs[modality], gen_targets[modality])
                    
                    batch_loss = batch_loss / len(gen_modalities)
                    val_loss += batch_loss.item()
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