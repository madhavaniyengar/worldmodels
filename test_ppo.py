import gymnasium as gym
import numpy as np
import torch

from lib.agent_ppo import PPOAgent

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("Humanoid-v5", render_mode='rgb_array')
    obs_dim = env.observation_space.shape
    action_dim = env.action_space.shape

    agent = PPOAgent(obs_dim[0], action_dim[0]).to(device)
    agent.load_state_dict(torch.load("model.pt"))
    agent.eval()

    obs, _ = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    frames = []
    while not done:
        # Render the frame and capture it
        frame = env.render()
        frames.append(frame)
        # Sample an action
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.tensor(np.array([obs], dtype=np.float32), device=device))
        # Step the environment
        obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        total_reward += reward
        step_count += 1
        done = terminated or truncated
    
    # Save the video
    import cv2
    if frames:
        out = cv2.VideoWriter("test_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print("Video saved as test_output.mp4")
    
    print(f"Episode completed!")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    
    env.close()
