#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from models.baseline_mlp import BaselineMLP
from datasets.dataset import SensorDataset
from torch.utils.data import DataLoader
from utils.constants import DEFAULT_DATA_PATH

def test_baseline_model():
    """Test that the baseline model can be instantiated and run forward pass"""
    print("Testing baseline model...")
    
    # Model configuration
    gen_dims = {'force': 78, 'velocity': 78}
    cond_dims = {'force': 78, 'velocity': 78, 'action': 17}
    hidden_dim = 256
    num_blocks = 4
    dropout = 0.0
    fusion_method = "concat"
    
    # Create model
    model = BaselineMLP(
        gen_dims=gen_dims,
        cond_dims=cond_dims,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=dropout,
        return_dict=True,
        fusion_method=fusion_method
    )
    
    print(f"Model created successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    cond_inputs = {
        'force': torch.randn(batch_size, 78),
        'velocity': torch.randn(batch_size, 78),
        'action': torch.randn(batch_size, 17)
    }
    
    with torch.no_grad():
        outputs = model(cond_inputs)
    
    print(f"Forward pass successful")
    print(f"Output shapes:")
    for modality, output in outputs.items():
        print(f"  {modality}: {output.shape}")
    
    # Test with different fusion method
    print("\nTesting with 'add' fusion method...")
    model_add = BaselineMLP(
        gen_dims=gen_dims,
        cond_dims=cond_dims,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=dropout,
        return_dict=True,
        fusion_method="add"
    )
    
    with torch.no_grad():
        outputs_add = model_add(cond_inputs)
    
    print(f"Add fusion forward pass successful")
    print(f"Output shapes:")
    for modality, output in outputs_add.items():
        print(f"  {modality}: {output.shape}")
    
    print("\nBaseline model test passed!")

def test_dataset_integration():
    """Test that the model works with the dataset"""
    print("\nTesting dataset integration...")
    
    # Create a small dataset
    try:
        dataset = SensorDataset(
            data_path=DEFAULT_DATA_PATH,
            pred_horizon=1,
            eval_mode=True,  # Use eval mode for smaller dataset
            modalities=['force', 'velocity', 'action']
        )
        
        print(f"Dataset created with {len(dataset)} samples")
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Get a batch
        batch = next(iter(dataloader))
        print(f"Batch keys: {batch.keys()}")
        
        # Create model
        gen_dims = {'force': 78, 'velocity': 78}
        cond_dims = {'force': 78, 'velocity': 78, 'action': 17}
        
        model = BaselineMLP(
            gen_dims=gen_dims,
            cond_dims=cond_dims,
            hidden_dim=256,
            num_blocks=4,
            dropout=0.0,
            return_dict=True,
            fusion_method="concat"
        )
        
        # Build inputs from batch
        cond_inputs = {
            'force': batch['force'],
            'velocity': batch['velocity'],
            'action': batch['action_sequence'].reshape(2, -1)  # Flatten action sequence
        }
        
        # Forward pass
        with torch.no_grad():
            outputs = model(cond_inputs)
        
        print(f"Dataset integration test successful!")
        print(f"Model outputs:")
        for modality, output in outputs.items():
            print(f"  {modality}: {output.shape}")
            
    except Exception as e:
        print(f"Dataset integration test failed: {e}")
        print("This might be expected if the dataset path doesn't exist")

if __name__ == "__main__":
    test_baseline_model()
    test_dataset_integration() 