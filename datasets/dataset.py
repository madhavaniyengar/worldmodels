import torch
import numpy as np
import os
from transformers import AutoProcessor, CLIPModel
from PIL import Image
import concurrent.futures
from utils.constants import FORCE_IDXS, VELOCITY_IDXS, DEFAULT_DATA_PATH

class SensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, pred_horizon=1, eval_mode=False, load_embedding=False, modalities=None, normalize=True, use_memory_mapping=False):
        self.pred_horizon = pred_horizon
        self.load_embedding = load_embedding
        self.normalize = normalize
        self.use_memory_mapping = use_memory_mapping
        
        if load_embedding:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to("cuda")
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        if modalities is None:
            self.modalities = ['image', 'force', 'state']
        else:
            self.modalities = modalities
            
        valid_modalities = ['image', 'force', 'velocity', 'state', 'action']
        for modality in self.modalities:
            if modality not in valid_modalities:
                raise ValueError(f"Invalid modality '{modality}'. Valid options are: {valid_modalities}")
        
        self.memory_mapped_data = {}
        self.episode_lengths = []
        
        episode_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        
        if eval_mode: 
            episode_dirs = episode_dirs[:10]
        else:
            pass

        self._init_data_loading(episode_dirs, data_path)
        
        self.norm_params = {}
        if self.normalize:
            self._init_normalization(data_path)
        
        self.all_samples = []
        for episode_idx, ep_len in enumerate(self.episode_lengths):
            for data_idx in range(ep_len - pred_horizon):
                self.all_samples.append((episode_idx, data_idx))
        self.all_samples = np.array(self.all_samples)

    def get_clip_embeddings(self, image_batch_np):
        pil_images = [Image.fromarray(img.astype('uint8')) for img in image_batch_np]
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            return self.model.get_image_features(**inputs)
    
    def _init_data_loading(self, episode_dirs, data_path):
        modalities_to_load = set(self.modalities)
        if 'force' in self.modalities or 'velocity' in self.modalities:
            modalities_to_load.add('state')
        
        # these are derived from state so we don't need to load them from disk
        modalities_to_load.discard('force')
        modalities_to_load.discard('velocity')
        
        print(f"Requested modalities: {self.modalities}")
        print(f"Modalities to load from disk: {modalities_to_load}")
        
        for modality in modalities_to_load:
            self.memory_mapped_data[modality] = []
            
        for episode_dir in episode_dirs:
            episode_path = os.path.join(data_path, episode_dir)
            
            episode_data = {}
            episode_length = None
            
            for modality in modalities_to_load:
                modality_file = os.path.join(episode_path, f"{modality}.npy")
                
                if os.path.exists(modality_file):
                    if self.use_memory_mapping:
                        modality_data = np.load(modality_file, mmap_mode='r')
                    else:
                        modality_data = np.load(modality_file)
                    
                    episode_data[modality] = modality_data
                    
                    if episode_length is None:
                        episode_length = modality_data.shape[0]
                    elif modality_data.shape[0] != episode_length:
                        raise ValueError(f"Inconsistent episode lengths: {modality} has {modality_data.shape[0]} timesteps, expected {episode_length}")
                        
                else:
                    print(f'Warning: {modality_file} not found for episode {episode_dir}')
                    episode_data[modality] = None
            
            for modality in modalities_to_load:
                self.memory_mapped_data[modality].append(episode_data[modality])
            
            if episode_length is not None:
                self.episode_lengths.append(episode_length)
            else:
                print(f"Warning: No valid data found for episode {episode_dir}")

    def _init_normalization(self, data_path):
        norm_file = os.path.join(data_path, "normalization_params.npz")
        
        if os.path.exists(norm_file):
            print("Loading existing normalization parameters")
            self._load_normalization_params(norm_file)
        else:
            print("Computing normalization parameters")
            self._compute_normalization_params()
            self._save_normalization_params(norm_file)

    def _compute_normalization_params(self):
        print("Computing normalization statistics")
        
        for modality in self.modalities:
            if modality in ['image']:
                continue
                
            print(f"Computing stats for {modality}")
            
            # collect all data for this modality
            all_data = []
            
            for episode_idx in range(len(self.episode_lengths)):
                if modality == 'force' or modality == 'velocity':
                    if 'state' in self.memory_mapped_data and self.memory_mapped_data['state'][episode_idx] is not None:
                        state_data = self.memory_mapped_data['state'][episode_idx]
                        for i in range(len(state_data)):
                            extracted = self._extract_from_state(state_data[i], modality)
                            all_data.append(extracted)
                else:
                    if modality in self.memory_mapped_data and self.memory_mapped_data[modality][episode_idx] is not None:
                        modality_data = self.memory_mapped_data[modality][episode_idx]
                        all_data.extend(modality_data)
            
            if all_data:
                all_data = np.array(all_data)
                mean = np.mean(all_data, axis=0)
                std = np.std(all_data, axis=0)
                std = np.where(std == 0, 1.0, std)
                
                self.norm_params[modality] = {
                    'mean': mean,
                    'std': std
                }
                print(f"{modality}: mean shape {mean.shape}, std shape {std.shape}")
            else:
                print(f"Warning: No data found for {modality}")

    def _save_normalization_params(self, norm_file):
        save_dict = {}
        for modality, params in self.norm_params.items():
            save_dict[f'{modality}_mean'] = params['mean']
            save_dict[f'{modality}_std'] = params['std']
        
        np.savez(norm_file, **save_dict)
        print(f"Saved normalization parameters to {norm_file}")

    def _load_normalization_params(self, norm_file):
        loaded = np.load(norm_file)
        
        for modality in self.modalities:
            if modality in ['image']:
                continue
                
            mean_key = f'{modality}_mean'
            std_key = f'{modality}_std'
            
            if mean_key in loaded and std_key in loaded:
                self.norm_params[modality] = {
                    'mean': loaded[mean_key],
                    'std': loaded[std_key]
                }
                print(f"Loaded {modality}: mean shape {self.norm_params[modality]['mean'].shape}")

    def _normalize_data(self, data, modality):
        if not self.normalize or modality not in self.norm_params or modality in ['image']:
            return data
        
        params = self.norm_params[modality]
        return (data - params['mean']) / params['std']

    def _extract_from_state(self, state, modality):
        if not isinstance(state, np.ndarray):
            state = np.array(state)
            
        if modality == 'force':
            return state[..., FORCE_IDXS[0]:]
        elif modality == 'velocity':
            return state[..., VELOCITY_IDXS[0]:VELOCITY_IDXS[1]]
        else:
            raise ValueError(f"Unknown modality for extraction: {modality}")

    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        episode_idx, data_idx = self.all_samples[idx]
        
        item = {}
        
        if 'image' in self.modalities and self.memory_mapped_data['image'][episode_idx] is not None:
            image = self.memory_mapped_data['image'][episode_idx][data_idx]
            if self.load_embedding:
                pil_image = Image.fromarray(image.astype('uint8'))
                inputs = self.processor(images=pil_image, return_tensors="pt", padding=True).to("cuda")
                with torch.no_grad():
                    image = self.model.get_image_features(**inputs).squeeze(0).cpu()
            else:
                image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            item['image'] = image
            
        if 'state' in self.modalities and self.memory_mapped_data['state'][episode_idx] is not None:
            state = self.memory_mapped_data['state'][episode_idx][data_idx]
            state = self._normalize_data(state, 'state')
            item['state'] = torch.tensor(state, dtype=torch.float32)
            
            target_state = self.memory_mapped_data['state'][episode_idx][data_idx + 1 : data_idx + 1 + self.pred_horizon]
            target_state = self._normalize_data(target_state, 'state')
            item['target_state'] = torch.tensor(target_state, dtype=torch.float32)
            
        if 'action' in self.modalities and self.memory_mapped_data['action'][episode_idx] is not None:
            action = self.memory_mapped_data['action'][episode_idx][data_idx]
            action = self._normalize_data(action, 'action')
            item['action'] = torch.tensor(action, dtype=torch.float32)
            
            action_sequence = self.memory_mapped_data['action'][episode_idx][data_idx + 1 : data_idx + 1 + self.pred_horizon]
            action_sequence = self._normalize_data(action_sequence, 'action')
            
            item['action_sequence'] = torch.tensor(action_sequence, dtype=torch.float32)
            
            
        if 'force' in self.modalities and 'state' in self.memory_mapped_data and self.memory_mapped_data['state'][episode_idx] is not None:
            state = np.array(self.memory_mapped_data['state'][episode_idx][data_idx])
            force = self._extract_from_state(state, 'force')
            force = self._normalize_data(force, 'force')
            item['force'] = torch.tensor(force, dtype=torch.float32)
            
            next_n_state = self.memory_mapped_data['state'][episode_idx][data_idx + 1 : data_idx + 1 + self.pred_horizon]
            target_force = self._extract_from_state(next_n_state, 'force')
            target_force = self._normalize_data(target_force, 'force')
            item['target_force'] = torch.tensor(target_force, dtype=torch.float32)
            
        if 'velocity' in self.modalities and 'state' in self.memory_mapped_data and self.memory_mapped_data['state'][episode_idx] is not None:
            state = np.array(self.memory_mapped_data['state'][episode_idx][data_idx])
            velocity = self._extract_from_state(state, 'velocity')
            velocity = self._normalize_data(velocity, 'velocity')
            item['velocity'] = torch.tensor(velocity, dtype=torch.float32)
            
            next_n_state = self.memory_mapped_data['state'][episode_idx][data_idx + 1 : data_idx + 1 + self.pred_horizon]
            target_velocity = self._extract_from_state(next_n_state, 'velocity')
            target_velocity = self._normalize_data(target_velocity, 'velocity')
            item['target_velocity'] = torch.tensor(target_velocity, dtype=torch.float32)
            
        return item
    
if __name__ == "__main__":
    dataset_full = SensorDataset(
        data_path=DEFAULT_DATA_PATH, 
        pred_horizon=10,
        modalities=['force', 'action', 'velocity'],
        eval_mode=True,
        use_memory_mapping=False,
        normalize=True,
    )
    print(len(dataset_full))
    
    loader = torch.utils.data.DataLoader(dataset_full, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    print("Dataset with normalization:")
    print(batch.keys())
    for key, value in batch.items():
        print(f'{key}: {value.shape}')
        if key in ['force', 'action', 'velocity', 'state']:
            print(f'{key} mean: {value.mean().item()}, std: {value.std().item()}')