import gymnasium as gym
import numpy as np
import torch
import os

from lib.agent_ppo import PPOAgent
from utils.constants import (
    DEFAULT_DATA_PATH,
    DEFAULT_PPO_MODEL_PATH,
)


def save_data(data, path):
    # save data as numpy
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.array(data))

if __name__ == "__main__":
    data_path = DEFAULT_DATA_PATH
    num_episodes = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("Humanoid-v5", render_mode='rgb_array')
    obs_dim = env.observation_space.shape
    action_dim = env.action_space.shape

    agent = PPOAgent(obs_dim[0], action_dim[0]).to(device)
    agent.load_state_dict(torch.load(DEFAULT_PPO_MODEL_PATH))
    agent.eval()
     
    

    for episode_idx in range(num_episodes):
        obs, _ = env.reset()
        done = False
        image_data = []
        cfrc_data = []
        state_data = []
        action_data = []
        velocity_data = []
        while not done:
            
            # Sample an action
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(torch.tensor(np.array([obs], dtype=np.float32), device=device))
                    
            # Step the environment
            obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            
            # extract image data
            # image_data.append(env.render())
            # extract contact force data
            cfrc_data.append(obs[-78:])
            # extract velocity data
            velocity_data.append(obs[175:253])
            # extract state data
            state_data.append(obs)
            # add action data
            action_data.append(action.cpu().numpy().squeeze(0))
            
            breakpoint()

            done = terminated or truncated
            
        data = {
            'image': image_data,
            'force': cfrc_data,
            'state': state_data,
            'action': action_data,
            'velocity': velocity_data,
        }
        # breakpoint()
        save_data(data, os.path.join(data_path, f'episode_{episode_idx}.npy'))
        print(f'Saved data for episode {episode_idx}')
        
        
        
    env.close()
