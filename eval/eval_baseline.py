import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.baseline_mlp import BaselineMLP
from datasets.dataset import SensorDataset
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

def load_model_from_checkpoint(checkpoint_path, gen_dims, cond_dims, hidden_dim, num_layers, dropout, fusion_method, device):
    model = BaselineMLP(
        gen_dims=gen_dims,
        cond_dims=cond_dims,
        hidden_dim=hidden_dim,
        num_blocks=num_layers,
        dropout=dropout,
        return_dict=True,
        fusion_method=fusion_method
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")
    return model

def evaluate_model(model, test_dataloader, gen_modalities, cond_modalities, future_cond_modalities, device):
    model.eval()
    total_loss = 0.0
    modality_losses = {mod: 0.0 for mod in gen_modalities}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            batch_size = batch[list(batch.keys())[0]].shape[0]
            
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
            
            batch_loss = 0.0
            for modality in gen_modalities:
                modality_loss = F.mse_loss(model_outputs[modality], gen_targets[modality])
                modality_losses[modality] += modality_loss.item()
                batch_loss += modality_loss
            
            total_loss += batch_loss.item()
            num_batches += 1
    
    avg_total_loss = total_loss / num_batches
    avg_modality_losses = {mod: loss / num_batches for mod, loss in modality_losses.items()}
    
    return avg_total_loss, avg_modality_losses

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

    batch_size = cfg.batch_size
    pred_horizon = cfg.pred_horizon
    force_dim = cfg.force_dim
    velocity_dim = cfg.velocity_dim
    action_dim = cfg.action_dim * pred_horizon
    state_dim = cfg.state_dim
    hidden_dim = cfg.hidden_dim
    num_layers = cfg.num_layers
    dropout = cfg.dropout
    fusion_method = cfg.get('fusion_method', 'concat')
    device = cfg.device if torch.cuda.is_available() else 'cpu'

    cond_modalities = cfg.get('cond_modalities', ['force', 'velocity', 'action'])
    gen_modalities = cfg.get('gen_modalities', ['force', 'velocity'])
    
    future_cond_modalities = cfg.get('future_cond_modalities', [])

    print(OmegaConf.to_yaml(cfg))

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

    device = torch.device(device)
    
    model = load_model_from_checkpoint(
        checkpoint_path, gen_dims, cond_dims, hidden_dim, num_layers, dropout, fusion_method, device
    )
    
    all_modalities = list(set(cond_modalities + gen_modalities))
    test_dataset = SensorDataset(
        data_path=cfg.data_path,
        pred_horizon=pred_horizon,
        eval_mode=True,
        modalities=all_modalities
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    print(f"Test samples: {len(test_dataset)}")
    
    avg_total_loss, avg_modality_losses = evaluate_model(
        model, test_dataloader, gen_modalities, cond_modalities, future_cond_modalities, device
    )
    
    print(f"avg loss: {avg_total_loss:.6f}")
    for modality, loss in avg_modality_losses.items():
        print(f"  {modality}: {loss:.6f}")

if __name__ == "__main__":
    main() 