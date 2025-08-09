import os
import numpy as np
from utils.constants import DEFAULT_DATA_PATH

def convert_dataset(old_data_path, new_data_path):
    """
    Converts episodes from data_path/episode_0.npy (with dict)
    into data_path/episode_0/{image.npy, state.npy, action.npy}
    """

    # Create new dataset folder
    os.makedirs(new_data_path, exist_ok=True)

    # Find all .npy files that store episode dicts
    npy_files = sorted([f for f in os.listdir(old_data_path) if f.endswith('.npy')])
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {old_data_path}")

    print(f"Found {len(npy_files)} episodes to convert.")

    for i, npy_file in enumerate(npy_files):
        episode_path = os.path.join(old_data_path, npy_file)
        episode_name = os.path.splitext(npy_file)[0]
        new_episode_dir = os.path.join(new_data_path, episode_name)

        # Create episode directory
        os.makedirs(new_episode_dir, exist_ok=True)

        print(f"[{i+1}/{len(npy_files)}] Converting {npy_file} -> {new_episode_dir}")

        # Load episode dict
        data = np.load(episode_path, allow_pickle=True).item()
        if not isinstance(data, dict):
            raise ValueError(f"{npy_file} does not contain a dict.")

        # Save each modality as a separate .npy file
        for key, value in data.items():
            key_file = os.path.join(new_episode_dir, f"{key}.npy")
            np.save(key_file, value)
            print(f"  Saved {key}.npy: shape {np.array(value).shape}")

    print("Conversion complete!")

if __name__ == "__main__":
    # Example usage
    old_data_path = str((DEFAULT_DATA_PATH.rstrip("/") + "_raw"))
    new_data_path = DEFAULT_DATA_PATH

    convert_dataset(old_data_path, new_data_path)
