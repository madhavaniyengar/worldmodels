# World Models

This is an initial look into a possible direction for studying generative world models for robotics.

Here we look at generating future partial observations given current/past observations. Specifically, we consider force and velocity in a MuJoCo humanoid environment.

## Dataset Generation
Data is collected using an expert policy, with some code/weights borrowed from https://github.com/ProfessorNova/PPO-Humanoid.
Run collect_sensor_data.py for data collection and then convert_dataset_structure.py for reorganizing dataset for easier loading. 

TODOs
- Expand the data collection to include noisy expert and random policy data.
- Expand to multiple tasks (Meta World is probably best for this)

## Training
I implemented two baselines: diffusion using an MLP denoiser (models/diffusion_lowdim.py) and a baseline residual MLP (baseline_mlp.py).
You can train them using the respective scripts in train/.

TODOs
- Improve the diffusion baseline by adding FiLM conditioning for the timestep

## Evaluation
You can find evaluation scripts in eval/ and visualize results using visualization/

I will try to update the documentation better when I have time. If you have questions feel free to email or slack me!
