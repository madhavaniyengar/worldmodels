import os
from pathlib import Path

# indices to read force and velocity from state
FORCE_IDXS = (-78, 0)
VELOCITY_IDXS = (175, 253)

REPO_ROOT: None

DEFAULT_OUTPUTS_DIR = None

DEFAULT_DATA_PATH = None

DEFAULT_PPO_MODEL_PATH = None
