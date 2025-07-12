from dataclasses import dataclass
import torch
import json
import os

# ANSI color codes
QUICK_COLOR = "\033[34m"  # Dark blue
REFERENCE_COLOR = "\033[31m"  # Dark red
UNDERLINE = "\033[4m"  # Underline
RESET = "\033[0m"  # Reset all formatting

def _load_model_dict():
    """Load model configurations from JSON file."""
    config_path = os.path.join(os.path.dirname(__file__), 'model_configs.json')
    with open(config_path, 'r') as f:
        return json.load(f)

# Model configurations
MODEL_DICT = _load_model_dict()

TOTAL_GPU_NUM = 2

@dataclass
class TOKEN:
    MATCH = 0
    # mismatch includes both neutral and divergent
    NEUTRAL = 1
    DIVERGENT = 2

@dataclass
class TOKEN_TYPE:
    INPUT_INSTRUCTION = 0
    REASONING = 1
    RESPONSE = 2